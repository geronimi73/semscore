from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
import torch, gc
import torch.nn as nn

class EmbeddingModelWrapper():
    DEFAULT_MODEL="sentence-transformers/all-mpnet-base-v2"

    def __init__(self, model_path=None, bs=8):
        if model_path is None: model_path = self.DEFAULT_MODEL
        self.model, self.tokenizer = self.load_model(model_path)
        self.bs = bs
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def load_model(self, model_path):
        model = AutoModel.from_pretrained(
            model_path,
        ).cuda()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
             model_path,
        )
        return model, tokenizer

    def emb_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences):
        embeddings=torch.tensor([],device="cuda")
        
        if self.bs is None:
            batches=[sentences]
        else:
            batches = [sentences[i:i + self.bs] for i in range(0, len(sentences), self.bs)]  
            
        for sentences in batches:
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to("cuda")
            with torch.no_grad():
                model_output = self.model(**encoded_input)        
            batch_embeddings=self.emb_mean_pooling(model_output, encoded_input['attention_mask'])
            
            embeddings=torch.cat( (embeddings, batch_embeddings), dim=0 )

        return embeddings

    def get_similarities(self, x, y=None):
        if y is None:
            num_samples=x.shape[0]
            similarities = [[0 for i in range(num_samples)] for f in range(num_samples)]
            for row in tqdm(range(num_samples)):
                similarities[row][0:row+1]=em.cos(x[row].repeat(row+1,1), x[0:row+1]).tolist()
            return similarities
        else:            
            return self.cos(x,y).tolist()

class ModelPredictionGenerator:
    def __init__(self, model, tokenizer, eval_dataset, use_accelerate=False, bs=8, generation_config=None):
        self.model=model
        self.tokenizer=tokenizer
        self.bs=bs
        self.eval_prompts=self.messages_to_prompts( eval_dataset )
        self.use_accelerate=use_accelerate
        self.accelerator = Accelerator()

        assert tokenizer.eos_token_id is not None
        assert tokenizer.chat_template is not None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # llama-precise
        if generation_config is None:            
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.1,
                "repetition_penalty": 1.18,
                "top_k": 40,
                "do_sample": True,
                "max_new_tokens": 100,
                "pad_token_id": tokenizer.pad_token_id
            }
        else:
            self.generation_config = generation_config

    def clear_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def messages_to_prompts(self, ds):
        prompts=[]
        for conversation in ds["messages"]:
            for i,msg in enumerate(conversation):
                if msg["role"]=="user":
                    prompts.append(
                        dict (
                            # prompt: format current messages up to the current user message and add a generation prompt
                            prompt=self.tokenizer.apply_chat_template(conversation[:i+1], add_generation_prompt=True, tokenize=False),
                            answer_ref=conversation[i+1]["content"]
                        )
                    )
        return prompts

    def get_batches(self, dataset, batch_size):
        return [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]  

    def tokenize_batch(self, batch):
        pad_side=self.tokenizer.padding_side
        self.tokenizer.padding_side="left"     # left pad for inference
        
        prompts=[ item["prompt"] for item in batch ]   
        prompts_tok=self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding='longest', 
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_length=True,
            pad_to_multiple_of=8,
            add_special_tokens=False
        ).to(self.model.device)
        self.tokenizer.padding_side=pad_side   # restore orig. padding side
        
        return prompts_tok

    def generate_batch(self, batch_tok):
        with torch.no_grad():
            outputs_tok=self.model.generate(
                input_ids=batch_tok["input_ids"],
                attention_mask=batch_tok["attention_mask"],
                **self.generation_config
            ).to("cpu")
        outputs=[
            # cut prompt from output
            self.tokenizer.decode(
                outputs_tok[i][outputs_tok[i] != self.tokenizer.pad_token_id][batch_tok["length"][i]:], 
                spaces_between_special_tokens=False,
                skip_special_tokens=True
                ).strip()
            for i,t in enumerate(outputs_tok) ]

        return outputs

    def run(self):
        self.model.eval()
        self.clear_cache()
    
        if self.use_accelerate:
            with self.accelerator.split_between_processes(list(range(len(self.eval_prompts)))) as eval_prompts_local_idcs:
                eval_prompts_local = [self.eval_prompts[i] for i in eval_prompts_local_idcs]
        else:
            eval_prompts_local = self.eval_prompts

        for batch in tqdm( self.get_batches(eval_prompts_local, self.bs) ):
            batch_tok = self.tokenize_batch( batch )
            answers = self.generate_batch( batch_tok )   
    
            for i in range(len(batch)):
                batch[i]["answer_pred"]=answers[i]
                batch[i]["GPU"]=self.accelerator.process_index
            
        if self.use_accelerate:
            return gather_object(eval_prompts_local)
        else:
            return eval_prompts_local