import torchvision
import torchvision.transforms as transforms
import torch
from PIL import Image
from os.path import join
from typing import Any, Callable, List, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, list_dir, list_files


class MyDataLoader(object):
    def __init__(self, choose=1):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Normalize((0.5,), (0.5,))])
        # 这里将background和eval所有类合并到一起，取每个字符的部分作为训练集，其他作为测试集
        # 964类->1623类
        self.trainloader = None
        # 659类->1623类
        self.testloader = None

        self.get_omniglot(choose)
        pass

    def get_omniglot(self, choose):
        if choose == 1:
            # trainset = torchvision.datasets.Omniglot(root='../omniglot/python/', background=True,
            #                                          download=True, transform=self.transform)
            trainset = MyDataSet(root='../omniglot/python/', background=True,
                                 download=True, transform=self.transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                           shuffle=True, num_workers=4)
        elif choose == 2:
            testset = torchvision.datasets.Omniglot(root='../omniglot/python/', background=False,
                                                    download=True, transform=self.transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                          shuffle=False, num_workers=4)
        elif choose == 3:
            # trainset = torchvision.datasets.Omniglot(root='../omniglot/python/', background=True,
            #                                          download=True, transform=self.transform)
            trainset = MyDataSet(root='../omniglot/python/', background=True,
                                 download=True, transform=self.transform)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                           shuffle=True, num_workers=4)
            # testset = torchvision.datasets.Omniglot(root='../omniglot/python/', background=False,
            #                                         download=True, transform=self.transform)
            testset = MyDataSet(root='../omniglot/python/', background=False,
                                download=True, transform=self.transform)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                          shuffle=False, num_workers=4)


class MyDataSet(VisionDataset):
    folder = ''

    def __init__(
            self,
            root: str,
            background: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(MyDataSet, self).__init__(join(root, self.folder), transform=transform,
                                        target_transform=target_transform)
        self.background = background

        # if download:
        #     self.download()
        #
        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        self.target_folder = join(self.root, "images_background_all")
        # self.target_folder_test = join(self.root, "images_evaluation")
        self._alphabets = list_dir(self.target_folder)
        # self._alphabets_test = list_dir(self.target_folder_test)
        # self._alphabets.extend(self._alphabets_test)
        self._characters: List[str] = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                           for a in self._alphabets], [])
        self._character_images = []
        # self._character_images = [
        #     [(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
        #     for idx, character in enumerate(self._characters)]
        if self.background:
            self._character_images = [
                [(image, idx) for imid, image in enumerate(list_files(join(self.target_folder, character), '.png'))
                 if imid < 10] for idx, character in enumerate(self._characters)]
        else:
            self._character_images = [
                [(image, idx) for imid, image in enumerate(list_files(join(self.target_folder, character), '.png'))
                 if imid >= 10] for idx, character in enumerate(self._characters)]
        self._flat_character_images: List[Tuple[str, int]] = sum(self._character_images, [])

    def __len__(self) -> int:
        return len(self._flat_character_images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, character_class = self._flat_character_images[index]
        image_path = join(self.target_folder, self._characters[character_class], image_name)
        image = Image.open(image_path, mode='r').convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _get_target_folder(self) -> str:
        return 'images_background' if self.background else 'images_evaluation'


if __name__ == '__main__':
    datas = MyDataLoader(3)
    trains = datas.trainloader
    tests = datas.testloader
    for i, data in enumerate(trains, 0):
        if i >= 3:
            break
        input, label = data
        print(input)
        print(label)

    print(len(trains))
    print(len(tests))
