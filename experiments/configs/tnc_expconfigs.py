from experiments.configs.base_expconfig import Base_ExpConfig

alltnc_expconfigs = {}

class TNC_ExpConfig(Base_ExpConfig):
    def __init__(self, w=0.2, mc_sample_size=10, epsilon=3, adf=False, encoder_dims=320, encoder_type='TS2Vec', **kwargs):
        super(TNC_ExpConfig, self).__init__(model_type="TNC", **kwargs)
        self.w = w
        self.mc_sample_size = mc_sample_size
        self.epsilon = epsilon
        self.adf = adf
        self.encoder_dims = encoder_dims
        self.encoder_type = encoder_type

#  variants for TNC-HAR experiments
epochs = 100
batch_size = 16

alltnc_expconfigs["tnc_har_TS2Vec_sim_0.2"] = TNC_ExpConfig(
    w=0.2, mc_sample_size=5, subseq_size=128, data_name="har", 
    epochs=epochs, lr=0.00001, batch_size=batch_size, save_epochfreq=10, 
    encoder_dims=320, encoder_type="TS2Vec", adf=False
)

alltnc_expconfigs["tnc_har_TS2Vec_sim_0.05"] = TNC_ExpConfig(
    w=0.05, mc_sample_size=5, subseq_size=128, data_name="har", 
    epochs=epochs, lr=0.00001, batch_size=batch_size, save_epochfreq=10, 
    encoder_dims=320, encoder_type="TS2Vec", adf=False
)

alltnc_expconfigs["tnc_har_TS2Vec_adf_0.2"] = TNC_ExpConfig(
    w=0.2, mc_sample_size=5, subseq_size=128, data_name="har", 
    epochs=epochs, lr=0.00001, batch_size=batch_size, save_epochfreq=10, 
    encoder_dims=320, encoder_type="TS2Vec", adf=True
)

alltnc_expconfigs["tnc_har_TS2Vec_adf_0.05"] = TNC_ExpConfig(
    w=0.05, mc_sample_size=5, subseq_size=128, data_name="har", 
    epochs=epochs, lr=0.00001, batch_size=batch_size, save_epochfreq=10, 
    encoder_dims=320, encoder_type="TS2Vec", adf=True
)

alltnc_expconfigs["tnc_har_RNN_sim_0.2"] = TNC_ExpConfig(
    w=0.2, mc_sample_size=5, subseq_size=128, data_name="har", 
    epochs=epochs, lr=0.00001, batch_size=batch_size, save_epochfreq=10, 
    encoder_dims=320, encoder_type="RNN", adf=False
)

alltnc_expconfigs["tnc_har_RNN_sim_0.05"] = TNC_ExpConfig(
    w=0.05, mc_sample_size=5, subseq_size=128, data_name="har", 
    epochs=epochs, lr=0.00001, batch_size=batch_size, save_epochfreq=10, 
    encoder_dims=320, encoder_type="RNN", adf=False
)

alltnc_expconfigs["tnc_har_RNN_adf_0.2"] = TNC_ExpConfig(
    w=0.2, mc_sample_size=5, subseq_size=128, data_name="har", 
    epochs=epochs, lr=0.00001, batch_size=batch_size, save_epochfreq=10, 
    encoder_dims=320, encoder_type="RNN", adf=True
)

alltnc_expconfigs["tnc_har_RNN_adf_0.05"] = TNC_ExpConfig(
    w=0.05, mc_sample_size=5, subseq_size=128, data_name="har", 
    epochs=epochs, lr=0.00001, batch_size=batch_size, save_epochfreq=10, 
    encoder_dims=320, encoder_type="RNN", adf=True
)
