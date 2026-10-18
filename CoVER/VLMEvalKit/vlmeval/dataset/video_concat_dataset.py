from ..smp import *
from .video_base import VideoBaseDataset


class ConcatVideoDataset(VideoBaseDataset):



    DATASET_SETS = {}

    def __init__(self, dataset, **kwargs):
        from . import build_dataset
        datasets = self.DATASET_SETS[dataset]
        self.dataset_map = {}

        self.dataset_name = dataset
        self.datasets = datasets
        self.nframe = kwargs.get('nframe', 0)
        self.fps = kwargs.get('fps', -1)
        for dname in datasets:
            dataset = build_dataset(dname, **kwargs)
            assert dataset is not None, dataset
            self.dataset_map[dname] = dataset
        TYPES = [x.TYPE for x in self.dataset_map.values()]
        MODALITIES = [x.MODALITY for x in self.dataset_map.values()]

        assert np.all([x == MODALITIES[0] for x in MODALITIES]), (datasets, MODALITIES)
        self.TYPE = TYPES
        self.MODALITY = MODALITIES[0]
        data_all = []
        for dname in datasets:
            data = self.dataset_map[dname].data
            data['SUB_DATASET'] = [dname] * len(data)
            data_all.append(data)

        data = pd.concat(data_all)
        data['original_index'] = data.pop('index')
        data['index'] = np.arange(len(data))
        self.data = data

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]
        idx = line['original_index']
        dname = line['SUB_DATASET']
        org_data = self.dataset_map[dname].data
        org_line = cp.deepcopy(org_data[org_data['index'] == idx]).iloc[0]
        return self.dataset_map[dname].build_prompt(org_line, video_llm)

    def dump_image(self, line):

        assert 'image' not in line
        assert 'image_path' in line
        tgt_path = toliststr(line['image_path'])
        return tgt_path

    @classmethod
    def supported_datasets(cls):
        return []

    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]

        data_all = load(eval_file)
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            data_sub = data_all[data_all['SUB_DATASET'] == dname]
            data_sub.pop('index')
            data_sub['index'] = data_sub.pop('original_index')
            data_sub.pop('SUB_DATASET')
            dump(data_sub, tgt)

        results_all = {}
        for dname in self.datasets:
            tgt = eval_file.replace(self.dataset_name, dname)
            res = self.dataset_map[dname].evaluate(tgt, **judge_kwargs)
            results_all.update(res)

        result = pd.DataFrame(results_all, index=['success', 'overall'])
        result = result.T
        for idx, item in result.iterrows():
            result.loc[idx, 'acc'] = round(item['success'] / item['overall'] * 100, 1)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(result, score_file)
        return result
