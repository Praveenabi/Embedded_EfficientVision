import timm
import torch
import onnx
from onnxsim import simplify

model_families = [
                     'repvit',
                     'fastvit',
                     'mobilevit',
                     'mobileone',
                     'efficientformer',
                     'efficientvit'
                     ]

model_names = []

for family in model_families:
    print(f"Model Family Name : {family}")
    model_names_list = timm.list_models(family+'*', pretrained=True)
    model_names_t  = [ model_name.split('.')[0]  for model_name in model_names_list]

    for model_name in model_names_t:
        if model_name not in model_names:
            model_names.append(model_name)

print("List of all pretrained model names :")
print(model_names)


for model_name in model_names:
    onnx_model_path = "onnx/" + model_name + ".onnx"
    print(f"onnx filename : {onnx_model_path}")

    #create model from timm
    model = timm.create_model(model_name, pretrained=True, exportable=True)
    model.eval()

    #create random input
    x = torch.randn(1,3,224,224, requires_grad=True)
    torch_out = model(x)

    #export to onnx format
    torch.onnx.export(model,
                    x,
                    onnx_model_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'])

    #check onnx model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    #check shape inference
    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)

    #simplify onnx model 
    model_simp, check = simplify(inferred_model)
    assert check, "Simplified Onnx model could not be validated"

    # save the simplified model
    onnx.save(model_simp, onnx_model_path)
    print(f"ONNX model simplified and saved to {onnx_model_path}")

