import pandas as pd
import torch
from torch.utils.data import Dataset

cols = ['A_t', 't', 'xt_gender', 'xt_hr', 'xt_sysbp', 'xt_diabp']


class ObservationalDataset(Dataset):
    def __init__(self, csv_file, columns=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        if columns is None:
            columns = cols
        data = pd.read_csv(csv_file)
        data_filtered = data[columns]
        self.ehr_data = []
        trajectory = []
        for i in range(len(data)):
            if i > 0 and data.iloc[i]['id'] == data.iloc[i-1]['id']:
                trajectory.append(torch.tensor(data_filtered.iloc[i].values))
            else:
                if i > 0:
                    self.ehr_data.append(torch.stack(trajectory))
                trajectory = [torch.tensor(data_filtered.iloc[i].values)]
        self.ehr_data.append(torch.stack(trajectory))

    def __len__(self):
        return len(self.ehr_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.ehr_data[idx]
        return sample
