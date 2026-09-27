from nnarchs.session import TensorRtSession
from nnarchs.model import TrtModel


def create_session(model: TrtModel) -> TensorRtSession:
    return model.framework.Session(model)

