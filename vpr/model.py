import torch
import torchvision

'''
our get_model_resnet50 function uses only the backbone with no aggregator.

This line model = torch.nn.Sequential(*list(model.children())[:-1])
strips the final fully-connected classification layer (model.fc), leaving you with:
Conv layers + BatchNorm + ReLU blocks → the backbone

AdaptiveAvgPool2d → this is the last remaining layer, which collapses the spatial feature map (B, 2048, H, W) down to (B, 2048, 1, 1)

That global average pooling is technically built into ResNet's architecture — it's not a standalone aggregator like NetVLAD, GeM,
or SALAD. It's just the backbone's own pooling head producing a single 2048-d vector per image.
So your pipeline is:
Image → ResNet50 backbone (conv layers) → AdaptiveAvgPool → 2048-d vector → L2 normalise

Compare this to a full VPR model which would look like:
Image → Backbone → Aggregator (GeM/NetVLAD/etc.) → descriptor

This has no explicit aggregator, which is fine for a baseline, but it's worth 
knowing that the built-in average pooling is generally weaker than a learned aggregator for place recognition tasks
'''
def get_model_resnet50(device="cpu"):
    model = torchvision.models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform