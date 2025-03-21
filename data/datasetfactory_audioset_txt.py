
import data.read_audioset_txt as read


train_path_phase0 = 'data_txt/train_phase_0.txt'
train_path_phase1 = 'data_txt/train_phase_1.txt'
train_path_phase2 = 'data_txt/train_phase_2.txt'
train_path_phase3 = 'data_txt/train_phase_3.txt'
train_path_phase4 = 'data_txt/train_phase_4.txt'
all_path_phase0_30 = 'data_txt/test_phase_0.txt'
all_path_phase1_35 = 'data_txt/test_phase_0_1.txt'
all_path_phase2_40 = 'data_txt/test_phase_0_1_2.txt'
all_path_phase3_45 = 'data_txt/test_phase_0_1_2_3.txt'
all_path_phase4_50 = 'data_txt/test_phase_0_1_2_3_4.txt'

class DatasetFactory:
    def __init__(self):
        pass
    @staticmethod
    def get_dataset(name, train=True, phase=0):
        if name == 'audioset':

            if train:
                if phase == 0:
                    return read.READ_DATA(train_path_phase0)
                elif phase == 1:
                    return read.READ_DATA(train_path_phase1)
                elif phase == 2:
                    return read.READ_DATA(train_path_phase2)
                elif phase == 3:
                    return read.READ_DATA(train_path_phase3)
                else:
                    return read.READ_DATA(train_path_phase4)

            else:
                if phase == 0:
                    return read.READ_DATA(all_path_phase0_30)
                elif phase == 1:
                    return read.READ_DATA(all_path_phase1_35)
                elif phase == 2:
                    return read.READ_DATA(all_path_phase2_40)
                elif phase == 3:
                    return read.READ_DATA(all_path_phase3_45)
                else:
                    return read.READ_DATA(all_path_phase4_50)

        else:
            print("Unsupported Dataset")
            assert False
