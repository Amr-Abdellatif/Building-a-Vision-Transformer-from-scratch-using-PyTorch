import torch
import torch.nn as nn
from ensure import ensure_annotations

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class patchembeddings(nn.Module):
    def __init__(self,
                in_channels:int = 3,
                patch_size:int =16,
                embdeddings_dim:int = 768):
        
        super(patchembeddings,self).__init__()
        """lets make a block called the patcher which is responsible for breaking
        the image into patches of patch_size """

        self.patch_size = patch_size

        self.patcher = nn.Conv2d(in_channels=in_channels,
                                out_channels=embdeddings_dim,
                                kernel_size=patch_size,
                                stride=patch_size,
                                padding=0
                                )
        
        self.flatten = nn.Flatten(start_dim=2,end_dim=3)
    
    @ensure_annotations
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # lets check for some condition
        image_res = x.shape[-1] # (1,224,224) -> 224
        assert image_res % self.patch_size == 0 , f"image size must be divisible by patch size, image_shape {x.shape}, patch size : {self.patch_size} "
        # patching
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # permute to return the right shape (1,196,768)
        return x_flattened.permute(0,2,1)
    

import torch
from torch import nn
class MultiHeadAttention(nn.Module):
    """
    creates a multi-head self-attention block 
    """
    def __init__(self,
                embedding_dim:int=768, # hidden size D for embeddings dim
                num_heads:int=12, # heads from table 1
                attn_dropout:int=0):
        super(MultiHeadAttention, self).__init__()
        # Create the Layer Norm
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create MultiHead self attention
        self.multi_head_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                num_heads=num_heads,
                                                dropout=attn_dropout,
                                                batch_first=True ) # batch first means the input is in shape of (batch,seq,feature)  -> (batch , number of patches , embedding dimension) 
    @ensure_annotations
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        attn_output , _ = self.multi_head_attn(query=x,
                                            key=x,
                                            value=x,
                                            need_weights=False)
        return attn_output
    
class MLPblock(nn.Module):
    def __init__(self,embedding_dim:int=768,
                mlp_size:int=3072,
                dropout:float=0.1):
        super(MLPblock, self).__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                    out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                    out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
    
    @ensure_annotations
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TranformerEncoderBlock(nn.Module):
    def __init__(self,
                embedding_dim:int=768,# hidden size D from table 1 , 768 for ViT-Base
                num_head:int=12, # from table 1
                mlp_size:int=3072 ,# from table 1
                mlp_dropout:int=0.1,# from table 3
                attn_dropout:int=0):
        super(TranformerEncoderBlock,self).__init__()

        # eqn 2
        self.msa_block = MultiHeadAttention(embedding_dim=embedding_dim,
                                            num_heads=num_head,
                                            attn_dropout=attn_dropout
                                            ) 
        # eqn 3
        self.mlp_block = MLPblock(embedding_dim=embedding_dim,
                                mlp_size=mlp_size,
                                dropout=mlp_dropout)
    
    @ensure_annotations
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = self.msa_block(x) + x # skip connection
        x = self.mlp_block(x) + x # skip connection
        return x
    

class ViT(nn.Module):
    def __init__(self,
                img_size:int=224, # table 3 from the paper
                in_channels:int = 3,
                patch_size:int=16,
                num_transformer_layer:int=12, # table 1 for layer ViT base
                embedding_dim:int =768, # hidden size D
                mlp_size:int=3072,
                num_heads:int = 12, # table 1
                attn_dropout:int=0,
                mlp_dropout:int=0.1,
                embedding_dropout:int=0.1, # dropout for patch and position embeddings
                num_classes:int=1000
                ):
        super(ViT,self).__init__()
        
        # make sure we have the compatible image size and patch size  
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image : {img_size} patch size {patch_size}"

        # calculate the number of patches (height*width) / patch_size**2
        self.num_patches = (img_size*img_size)// patch_size**2

        # create Learnable class embeddings
        """we have the shape of (1,196,768) we need to prepend class token -> (1,197,768)"""
        self.class_embeddings = nn.Parameter(data=torch.randn(1,1,embedding_dim),requires_grad=True)

        # create Learnable position embeddings
        self.position_embeddings = nn.Parameter(data=torch.randn(1,self.num_patches+1,embedding_dim),requires_grad=True)

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create patch embeddings 
        self.patch_embeddings = patchembeddings(in_channels=in_channels,
                                                embdeddings_dim=embedding_dim,
                                                patch_size=patch_size)
        
        """now to make the multi head attention we use the module layers list property in PyTorch"""
        self.transformer_encoder = nn.Sequential(*[TranformerEncoderBlock(embedding_dim=embedding_dim,
                                                                        num_head=num_heads,
                                                                        mlp_size=mlp_size,
                                                                        mlp_dropout=mlp_dropout)
                                                                        for _ in range(num_transformer_layer)])
        
        # Create Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,out_features=num_classes)
        ) 

    @ensure_annotations
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Get th patch size
        batch_size = x.shape[0]
        
        # Create class token embedding and expand it to match the batch size eqn 1
        class_token = self.class_embeddings.expand(batch_size,-1,-1) # -1 means to infer the dimensions

        # 14. Create patch embedding (equation 1)
        x = self.patch_embeddings(x)

        # concat class token and patch embeddings eqn 1
        x = torch.cat(tensors=(class_token,x),dim=1)

        # add position embeddings to class token and patch embeddings
        x = self.position_embeddings + x
        
        # Apply dropout to patch embeddings
        x = self.embedding_dropout(x)

        # pass position and patch embeddings to transformer encoder (eqn 2,3)
        x = self.transformer_encoder(x)

        # put the 0th index logit through the classifier
        x = self.classifier(x[:,0])

        return x        


