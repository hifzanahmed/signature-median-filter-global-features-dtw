from utilities import Utilities
from signature_feature_extraction import SignatureFeatureExtraction
import config

class SignatureVerificationTraining:
    def verifiy_test_signature(test_image_path):
        test_feature = SignatureFeatureExtraction.preprocess_and_feature_extraction(test_image_path)
        utils = Utilities()
        dtw_distance = utils.compute_verification_score(test_feature, config.global_features)
        #print("Computed DTW Distance:", dtw_distance)
        return dtw_distance