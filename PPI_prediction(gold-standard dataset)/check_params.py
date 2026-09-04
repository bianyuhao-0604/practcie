import torch
import models.attention as attention

# Load the pretrained model
model_path = 'bachelor/pytorchtest/models/pretrained/crossattention_esm2_t33_650.pt'
model = attention.CrossAttInteraction(1280, 8)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Count the number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total parameters: {total_params}')
print(f'Trainable parameters: {trainable_params}')