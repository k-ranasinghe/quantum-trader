import torch

from generators.signal_generator import SignalGenerator
from config.constants import CONSTANTS


def test_signal_generator_shapes():
    """
    Test that the SignalGenerator model outputs correct shapes
    """
    batch_size = 4
    seq_len = CONSTANTS.SEQ_LEN
    input_dim = CONSTANTS.INPUT_DIM
    
    model = SignalGenerator(input_dim=input_dim, seq_len=seq_len)
    
    # Create random input tensor [batch, seq_len, features]
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output = model(x)
    
    assert 'action_logits' in output
    assert output['action_logits'].shape == (batch_size, 2)
    
    assert 'entry_range' in output
    assert output['entry_range'].shape == (batch_size, 2)
    
    assert 'leverage' in output
    assert output['leverage'].shape == (batch_size, 1)
    
    assert 'confidence' in output
    assert output['confidence'].shape == (batch_size, 1)

def test_feature_columns_consistency():
    """
    Ensure feature columns in constant match model expectations
    """
    assert len(CONSTANTS.FEATURE_COLS) == CONSTANTS.INPUT_DIM
