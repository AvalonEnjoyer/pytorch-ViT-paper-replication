"""Class for creating a ViT model"""
import torch
from torch import nn

class PatchEmbedding(nn.Module):
  def __init__(self,
               image_size:int,
               color_channels:int=3,
               patch_size:int=16,
               embedding_dropout:float=0.1, # Dropout for patch and position embeddings
               embedding_dim:int=768,): # from Table1, defaults for a ViT-Base model
    """
    Returns the patch embedding for a given image in the shape of
    (batch_size, number_of_patches, embedding_dim)
    Image must be of square shape: H == W
    Args:
      image_size (int) - Image H / W, image_size % patch_size must be 0
      color_channels (int) - Number of color channels in the image
      patch_size (int) - Patch size, image_size % patch_size must be 0
      embedding_dropout (float), Dropout for patch and position embeddings
      embedding_dim (int) - Number of embedding dimensions for the vector
    """
    super().__init__()
    self.patch_size = patch_size
    assert image_size % self.patch_size == 0, f"Input image size must be divisible by patch size, image size: {image_size}, patch_size: {patch_size}"
    self.number_of_patches = int((image_size**2)/(patch_size**2)) # Assuming square image
    self.patch_and_flatten = nn.Sequential(
        nn.Conv2d(in_channels=color_channels,
                  out_channels=embedding_dim,
                  kernel_size=patch_size,
                  stride=patch_size),
        nn.Flatten(start_dim=2,
                              end_dim=3)
    )
    self.class_token = nn.Parameter(torch.rand(1,
                                               1,
                                               embedding_dim),
                                    requires_grad=True)
    self.position_embedding = nn.Parameter(torch.rand(1,
                                                      self.number_of_patches+1,
                                                      embedding_dim),
                                           requires_grad=True)
    self.dropout = nn.Dropout(p=embedding_dropout)

  def forward(self, x:torch.Tensor):
    # Get batch size
    batch_size = x.shape[0]

    # Create class token embedding and expand it to match the batch size (equation 1)
    class_token = self.class_token.expand(batch_size,-1,-1) # -1 means to infer the dimensions

    # Create the patch embedding (equation 1)
    x = self.patch_and_flatten(x)

    # Set the sequence embedding dimension in the right order(batch_size, number_of_patches, embedding_dimension)
    x = x.permute(0,2,1)

    # Prepend the class token
    x = torch.cat((class_token,x),
                  dim=1)

    # Add the position embedding
    x = x + self.position_embedding

    # Apply dropout to patch embedding ("directly after adding positional - to patch embeddings")
    x = self.dropout(x)
    return x

class MultiHeadSelfAttentionBlock(nn.Module):
  """Creates a mult-head self-attention block ("MSA block" for short)
  Args:
    embedding_dim (int) - Embedding dimensions, 768 by default for ViT-Base
    num_heads (int) - Number of heads for the mult-head self-attention
    attn_dropout (float) - Dropout rate for multihead self attention
  """
  def __init__(self,
               embedding_dim:int=768, # Hidden size D (embedding dimension) from Table 1 for ViT-Base
               num_heads:int=12, # Heads from Table 1 for ViT-Base
               attn_dropout:float=0):
    super().__init__()

    # Create the norm layer (LN)
    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    # Create multihead attention (MSA) layer
    self.multi_head_attention = nn.MultiheadAttention(
        embed_dim=embedding_dim,
        num_heads=num_heads,
        dropout=attn_dropout,
        batch_first=True) # is the batch first? (batch, seq, feature) -> (batch, number_of_patches, embedding_dimension)

  def forward(self, x:torch.Tensor):
    normalized_x = self.layer_norm(x)
    attn_output, _ = self.multi_head_attention(query=normalized_x,
                                               key=normalized_x,
                                               value=normalized_x,
                                               need_weights=False)
    return attn_output

class MLPBlock(nn.Module):
  """Takes the output of the MSA block and returns the prediction logits added
  to the input.
  Args:
    embedding_dim (int) = Embedding dimension, default 768 for ViT-Base
    mlp_size (int) = Features for MLP block, default 768 for ViT-Base
    dropout (float) = Dropout rate between 0 and 1, default 0.1 for ViT-Base
  """
  def __init__(self,
               embedding_dim:int=768,
               mlp_size=3072,
               dropout:float=0.1):
    super().__init__()

    # Create the layer norm layer (LN)
    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    # Create the MLP
    self.mlp = nn.Sequential(
        nn.Linear(in_features=embedding_dim,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,
                  out_features=embedding_dim),
        nn.Dropout(p=dropout)
    )

  def forward(self, x:torch.Tensor):
    normalized_x = self.layer_norm(x)
    mlp_x = self.mlp(normalized_x)
    return mlp_x

class TransformerEncoderBlock(nn.Module):
  def __init__(self,
               embedding_dim:int=768, # Hidden size D (embedding dimension) from Table 1 for ViT-Base
               num_heads:int=12, # Heads from Table 1 for ViT-Base
               attn_dropout:float=0,
               mlp_size=3072,
               mlp_dropout:float=0.1):
    """
    Passes embedded patches through a multihead self attention (MSA) block and
    multilayer perceptron (MLP) block
    Args:
      embedding_dim (int) - Embedding dimension for the vector, 769 for ViT-Base
      num_heads (int) - Number of heads for the mult-head self-attention
      attn_dropout (float) - Dropout rate for multihead self attention
      mlp_size (int) = Features for MLP block, default 768 for ViT-Base
      mlp_dropout (float) = Dropout rate between 0 and 1, default 0.1 for ViT-Base
    """
    super().__init__()

    # Create MSA block (equation 2)
    self.msa_block = MultiHeadSelfAttentionBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        attn_dropout=attn_dropout
    )

    # Create MLP block (equation 3)
    self.mlp_block = MLPBlock(
        embedding_dim=embedding_dim,
        mlp_size=mlp_size,
        dropout=mlp_dropout
    )

  def forward(self, x:torch.Tensor):
    x = self.msa_block(x) + x # residual/skip connection for equation 2
    x = self.mlp_block(x) + x # residual/skip connection for equation 3
    return x

# ViT model class
class ViT(nn.Module):
  def __init__(self,
               image_size:int=224, # Table 3 from the ViT paper
               patch_size:int=16,
               num_transformer_layer:int=12, # Table 1 for `Layer` for ViT-Base
               num_heads:int=12, # Table 1 for 'Heads' for ViT-Base
               embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
               mlp_size:int=3072,
               attn_dropout:float=0,
               mlp_dropout:float=0.1,
               embedding_dropout:int=0.1, # Dropout for patch and position embeddings
               num_classes:int=1000): # number of classes in our classification problem
    """
    Returns a ViT model based on A picture is worth 16x16 words, default values
    for ViT_Base model trained on ImageNet
    Args:
    image_size (int) = H/W dimension of the image
    patch_size (int) = H/W dimension for patch, image_size**2 // patch_size**2 must be 0
    embedding_dim (int) = Embedding dimension
    mlp_size (int) = number of nodes in the multi-layer perceptron (MLP)
    num_heads (int) = number of heads in the multihead self attention layer
    num_transformer_layer (int), number of Transformer Encoder layer in ViT-Base
    mlp_dropout (float) = dropout percentage for the MLP
    embedding_dropout (float) = dropout percentage for after embedding
    num_class (int) = number of classes in the data
    """
    super().__init__()

    # Create the patch embedding block
    self.patch_embedding_block = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        embedding_dim=embedding_dim,
        embedding_dropout=embedding_dropout)

    # Create the transformer encoder block
    self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        attn_dropout=attn_dropout,
        mlp_size=mlp_size,
        mlp_dropout=mlp_dropout
    ) for _ in range(num_transformer_layer)])

    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    # Create classifier head
    self.classifier = nn.Sequential(
        nn.Linear(in_features=embedding_dim,
                  out_features=num_classes)
    )

  def forward(self, x:torch.Tensor):
    # Get position and patch embedding (equation 1)
    x = self.patch_embedding_block(x)

    # Pass position and patch embedding through transformer encoder (equations 2 and 3)
    x = self.transformer_encoder(x)

    # Layer norm for the output of the transformer encoder
    x = self.layer_norm(x)

    # Put 0th index logit through the classifier
    x = self.classifier(x[:,0])
    return x
