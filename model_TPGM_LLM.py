import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from gcn import GCN


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        # Extract day and week info from input tensor x
        day_emb = x[..., 1]    
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)   
        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)
        tem_emb = time_day + time_week
        return tem_emb
    
class TextEmbeddingBERT(nn.Module):
    def __init__(self, device="cuda:0", bert_layers=2, hidden_size=768):
        super(TextEmbeddingBERT, self).__init__()
        self.device = device  # Save device
        self.hidden_size = hidden_size
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained("/root/autodl-tmp/model/bert-base-uncased")
        # Use only the first bert_layers
        self.bert.encoder.layer = self.bert.encoder.layer[:bert_layers]
        # Freeze all parameters in the BERT model
        for param in self.bert.parameters():
            param.requires_grad = False
        # Unfreeze only parts related to text embedding
        # 1. Unfreeze word_embeddings
        for param in self.bert.embeddings.word_embeddings.parameters():
            param.requires_grad = True
        # 2. Unfreeze position_embeddings
        for param in self.bert.embeddings.position_embeddings.parameters():
            param.requires_grad = True
        # 3. Unfreeze token_type_embeddings
        for param in self.bert.embeddings.token_type_embeddings.parameters():
            param.requires_grad = True
        # 4. Unfreeze LayerNorm
        for param in self.bert.embeddings.LayerNorm.parameters():
            param.requires_grad = True
        # 5. Unfreeze Dropout
        for param in self.bert.embeddings.dropout.parameters():
            param.requires_grad = True
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("/root/autodl-tmp/model/bert-base-uncased")
        # Check which parameters are frozen and which are trainable
        self.check_frozen_params()
    
    def check_frozen_params(self):
        """Check which parameters in the BERT model are frozen and which are trainable"""
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")
    
    def forward(self, input_text):
        device = self.device
        input_text = input_text.tolist()
        embeddings = []
        for i in range(len(input_text)):  # Iterate over each batch
            batch_embeddings = []
            for text in input_text[i]:  # Iterate over each text
                # Tokenize, ensuring all text sequence lengths are 512 (including padding and truncation)
                encoding = self.tokenizer(text, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                # Get BERT's embedding layer output
                with torch.no_grad():
                    word_embeds = self.bert.embeddings(input_ids)  # Shape (seq_len, hidden_size)
                # Encode via BERT
                outputs = self.bert(
                    inputs_embeds=word_embeds,
                    attention_mask=attention_mask
                )
                # Get BERT output embeddings, shape (batch_size, seq_len, hidden_size)
                embeddings_text = outputs.last_hidden_state.squeeze(0)  # Remove single batch dimension
                # Perform Z-score normalization on embeddings (mean=0, variance=1)
                mean = embeddings_text.mean(dim=-1, keepdim=True)  # Calculate mean for each feature
                std = embeddings_text.std(dim=-1, keepdim=True)   # Calculate standard deviation for each feature
                # Prevent division by zero by adding a very small constant eps
                eps = 1e-6
                embeddings_text = (embeddings_text - mean) / (std + eps)
                # Add normalized embeddings to the batch
                batch_embeddings.append(embeddings_text)
            # Stack embeddings of this batch into shape (15, 512, hidden_size)
            batch_embeddings = torch.stack(batch_embeddings, dim=0)
            embeddings.append(batch_embeddings)
        # Final output shape (batch_size, 15, 512, hidden_size)
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings  # Shape: (batch_size, 15, 512, hidden_size)


class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()
        
        self.gpt2 = GPT2Model.from_pretrained(
            "/root/autodl-tmp/model/gpt2_model", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state


class TPGM_LLM(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=1296,
        input_len=15,
        output_len=15,
        dropout=0.1,
        U=1,
        X=None,  # Add X
        H=None,  # Add H
        W=None   # Add W
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.U = U
        self.X = X
        self.H = H
        self.W = W
         
        time = 360

        gpt_channel = 256
        to_gpt_channel = 768

        self.Temb = TemporalEmbedding(time, gpt_channel)
        self.Textemb = TextEmbeddingBERT()

        self.gcn = GCN(input_dim=15, hidden_dim=128, output_dim=gpt_channel)

        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )
        # embedding layer
        self.gpt = PFA(device=self.device, gpt_layers=1, U=self.U)
        self.feature_fusion = nn.Conv2d(
            gpt_channel * 4, to_gpt_channel, kernel_size=(1, 1)
        )
        # regression
        self.regression_layer = nn.Conv2d(
            gpt_channel * 3, self.output_len, kernel_size=(1, 1)
        )

        # Expand 1 to 1296
        self.text_conv2 = nn.Conv2d(
            512, 768, kernel_size=(1, 1)
        )
        self.text_conv3 = nn.Conv2d(
            768, 1296, kernel_size=(1, 1)
        )
        self.text_start_conv = nn.Conv2d(
            768 * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        self.node_conv = nn.Conv2d(
            5, 1296, kernel_size=(1, 1)
        )

        # self.input_text_dim * self.input_len

    # Return the total number of model parameters
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data, text): 
        # Initial dimension of input data
        input_data = history_data
        batch_size, _, num_nodes, _ = input_data.shape

        node_emb = self.gcn(input_data)
        # print(f"Node_emb dimension after GCN convolution: {node_emb.shape}")  # Output shape after GCN convolution
        node_emb = node_emb.unsqueeze(-1)
       
        text_emb = self.Textemb(text)  # torch.Size([64, 15, 512, 768])
        # print(f"text_emb dimension after getting text embedding via Textemb: {text_emb.shape}")
        
        text_emb = text_emb.to(self.device)
        text_emb = text_emb.permute(0, 2, 1, 3)  # torch.Size([64, 1, 15, 768])
        # print(f"text_emb shape after permute(0, 2, 1, 3): {text_emb.shape}")
        
        # text_emb = self.text_conv1(text_emb)  # torch.Size([64, 1296, 15, 768])  # Commented out
        text_emb = self.text_conv2(text_emb)
        text_emb = self.text_conv3(text_emb)
        # print(f"text_emb dimension after text_conv2 and text_conv3: {text_emb.shape}")
        
        text_emb = text_emb.permute(0, 1, 3, 2)  # torch.Size([64, 1296, 768, 15])
        # print(f"text_emb shape after permute(0, 1, 3, 2): {text_emb.shape}")

        text_emb = text_emb.reshape(batch_size, 1296, -1).transpose(1, 2).unsqueeze(-1)  # torch.Size([64, 11520, 1296, 1])
        # print(f"text_emb shape after reshape and transpose: {text_emb.shape}")
        
        text_emb = self.text_start_conv(text_emb)  # torch.Size([64, 256, 1296, 1])
        # print(f"text_emb dimension after text_start_conv: {text_emb.shape}")

        history_data = history_data.permute(0, 3, 2, 1)  # Change data dimension
        # print(f"History data dimension after permute(0, 3, 2, 1): {history_data.shape}")# torch.Size([4, 15, 1296, 3])
        
        tem_emb = self.Temb(history_data)
        # print(f"tem_emb dimension obtained via Temb: {tem_emb.shape}")

        
        input_data = input_data.transpose(1, 2).contiguous()
        # print(f"Input data shape after transpose(1, 2): {input_data.shape}")
        
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        # print(f"Input data shape after view and transpose: {input_data.shape}")
        
        input_data = self.start_conv(input_data)
        # print(f"input_data dimension after start_conv: {input_data.shape}")
        # input_data = self.gcn(input_data)
        # input_data = input_data.unsqueeze(-1)
        # print(f"input_data dimension after GCN convolution: {input_data.shape}") #torch.Size([4, 1296, 512, 1])
        # input_data = input_data.transpose(1,2)
        
        # Concatenate different embeddings
        data_st = torch.cat([input_data] + [tem_emb] + [text_emb] + [node_emb], dim=1)  # torch.Size([64, 1024, 1296, 1])
        # print(f"Concatenated data_st dimension: {data_st.shape}")
        
        data_st = self.feature_fusion(data_st)
        # print(f"data_st dimension after feature_fusion: {data_st.shape}")

        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        # print(f"data_st shape after permute(0, 2, 1, 3) and squeeze: {data_st.shape}")

        sequence_length = data_st.shape[1]  # Get sequence length
        max_seq_len = 1024  # GPT2's maximum sequence length
        # print(f"Sequence length sequence_length: {sequence_length}, Maximum sequence length max_seq_len: {max_seq_len}")
        
        outputs = []
        for i in range(0, sequence_length, max_seq_len):
            chunk = data_st[:, i:i + max_seq_len, :]  # Split into multiple chunks
            # print(f"Processing chunk {i//max_seq_len + 1}, shape: {chunk.shape}")  # Print shape of each chunk
            output_chunk = self.gpt(chunk)  # Pass through GPT model chunk by chunk
            outputs.append(output_chunk)

        data_st = torch.cat(outputs, dim=1)
        #print(f"Merged data_st dimension: {data_st.shape}")

        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)
        # print(f"data_st shape after permute(0, 2, 1) and unsqueeze: {data_st.shape}")
        
        prediction = self.regression_layer(data_st)  # torch.Size([64, 15, 1296, 1])
        # print(f"Prediction result dimension after regression layer: {prediction.shape}")

        return prediction