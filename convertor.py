import torch
from model.ESPCN import ESPCN

def convertor():
    pytorch_model = ESPCN(n_colors=3, scale=2)
    pytorch_model.load_state_dict(torch.load('trained_model/ESPCN/ESPCN.pkl'))
    pytorch_model.eval()
    dummy_input = torch.zeros([1,3,640,480])
    torch.onnx.export(pytorch_model, dummy_input, 'ESPCN_ONNX.onnx', export_params=True, verbose=True)

if __name__ == '__main__':
    convertor()