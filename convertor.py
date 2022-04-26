ONNXPATH = 'ESPCN_ONNX.onnx'
TFPATH = 'tf_model/'
def convertor_pytorch2onnx():
    import torch
    from model.ESPCN import ESPCN
    device = 'cuda'
    pytorch_model = ESPCN(n_colors=3, scale=2)
    pytorch_model.load_state_dict(torch.load('trained_model/ESPCN/ESPCN.pkl'))
    pytorch_model.to(device)
    pytorch_model.eval()
    dummy_input = torch.zeros([1,3,640,480]).to(device)
    # torch.onnx.export(pytorch_model, dummy_input, 'ESPCN_ONNX.onnx', export_params=True, verbose=True)
    torch.onnx.export(pytorch_model, dummy_input, ONNXPATH, export_params=True, verbose=True, input_names=['input'], output_names=['output'])

# def convertor_onnx2tf():
#     import onnx
#     from onnx_tf.backend import prepare
#     onnx_model = onnx.load(ONNXPATH)
#     tf_model = prepare(onnx_model)
#     print(tf_model.summary())
#     # tf_model.export_graph(TFPATH)

# def convertor_pytorch2keras():
#     import torch
#     from model.ESPCN import ESPCN
#     from pytorch2keras.converter import pytorch_to_keras
#     pytorch_model = ESPCN(n_colors=3, scale=2)
#     pytorch_model.load_state_dict(torch.load('trained_model/ESPCN/ESPCN.pkl'))
#     pytorch_model.eval()
#     dummy_input = torch.zeros([1,3,480,640])
#     keras_model = pytorch_to_keras(pytorch_model, dummy_input)
#     print(keras_model.summary())

    


if __name__ == '__main__':
    convertor_pytorch2onnx()
    # convertor_onnx2tf()
    # convertor_pytorch2keras()
    pass