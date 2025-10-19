from multiprocessing import cpu_count
from torchvision import datasets, transforms
import torch
import multiprocessing


def get_dataloader(root_path, image_size, batch_size, workers=8):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_train = datasets.CIFAR10(
        root=root_path, download=True, train=True, transform=transform
    )

    dataset_test = datasets.CIFAR10(
        root=root_path, download=True, train=False, transform=transform
    )
    
    print(f"Using {workers} workers")
    
    # Extracting only the automobile indices (Label = 1)
    train_indices = [i for i, (_, label) in enumerate(dataset_train) if label == 1]
    test_indices = [i for i, (_, label) in enumerate(dataset_test) if label == 1]
    
    # # Create a subset using those indices
    train_car_subset = torch.utils.data.Subset(dataset_train, train_indices)
    test_car_subset = torch.utils.data.Subset(dataset_test, test_indices)

    dataset = torch.utils.data.ConcatDataset([train_car_subset, test_car_subset])
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers,
        pin_memory=True, 
        persistent_workers=True if workers > 0 else False
    )

    return dataloader

# def collate_fn(batch):
    
#     return (
#         torch.stack([x[0] for x in batch]), 
#         torch.tensor([x[1] for x in batch])
#     )


# def get_dataloader(root_path, image_size, batch_size, workers=multiprocessing.cpu_count()):
#     transform = transforms.Compose(
#         [
#             transforms.Resize(image_size),
#             transforms.CenterCrop((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#     )

#     dataset_train = datasets.StanfordCars(
#         root=root_path, download=False, split='train', transform=transform
#     )
    
#     dataset_test = datasets.StanfordCars(
#         root=root_path, download=False, split='test', transform=transform
#     )

#     dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
    
#     print(f"Using {workers} workers")
#     dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
#         pin_memory=True, 
#         persistent_workers=True if workers > 0 else False,
# #         collate_fn=collate_fn
#     )

#     return dataloader
