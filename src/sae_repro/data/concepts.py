from __future__ import annotations

import numpy as np


COARSE_TO_FINE: dict[str, tuple[str, ...]] = {
    "aquatic_mammals": ("beaver", "dolphin", "otter", "seal", "whale"),
    "fish": ("aquarium_fish", "flatfish", "ray", "shark", "trout"),
    "flowers": ("orchids", "poppies", "roses", "sunflowers", "tulips"),
    "food_containers": ("bottles", "bowls", "cans", "cups", "plates"),
    "fruit_and_vegetables": ("apples", "mushrooms", "oranges", "pears", "sweet_peppers"),
    "household_electrical_devices": ("clock", "computer_keyboard", "lamp", "telephone", "television"),
    "household_furniture": ("bed", "chair", "couch", "table", "wardrobe"),
    "insects": ("bee", "beetle", "butterfly", "caterpillar", "cockroach"),
    "large_carnivores": ("bear", "leopard", "lion", "tiger", "wolf"),
    "large_man_made_outdoor_things": ("bridge", "castle", "house", "road", "skyscraper"),
    "large_natural_outdoor_scenes": ("cloud", "forest", "mountain", "plain", "sea"),
    "large_omnivores_and_herbivores": ("camel", "cattle", "chimpanzee", "elephant", "kangaroo"),
    "medium_sized_mammals": ("fox", "porcupine", "possum", "raccoon", "skunk"),
    "non_insect_invertebrates": ("crab", "lobster", "snail", "spider", "worm"),
    "people": ("baby", "boy", "girl", "man", "woman"),
    "reptiles": ("crocodile", "dinosaur", "lizard", "snake", "turtle"),
    "small_mammals": ("hamster", "mouse", "rabbit", "shrew", "squirrel"),
    "trees": ("maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"),
    "vehicles_1": ("bicycle", "bus", "motorcycle", "pickup_truck", "train"),
    "vehicles_2": ("lawn_mower", "rocket", "streetcar", "tank", "tractor"),
}


def coarse_names() -> list[str]:
    """返回固定顺序的 20 个 coarse 类别名。"""
    return list(COARSE_TO_FINE)


def fine_to_coarse_map(fine_names: list[str]) -> np.ndarray:
    """依据官方类别名生成 fine label 到 coarse label 的映射。"""
    owner: dict[str, int] = {}
    for coarse_id, group_name in enumerate(coarse_names()):
        for fine_name in COARSE_TO_FINE[group_name]:
            owner[fine_name] = coarse_id
    missing = [name for name in fine_names if name not in owner]
    if missing:
        raise ValueError(f"存在未知 CIFAR-100 fine 类别：{missing}")
    return np.asarray([owner[name] for name in fine_names], dtype=np.int64)


def build_concept_matrix(
    fine_labels: np.ndarray,
    coarse_labels: np.ndarray,
    num_fine: int = 100,
    num_coarse: int = 20,
) -> np.ndarray:
    """构造每张图只激活 fine 和 coarse 两维的层级概念矩阵。"""
    if fine_labels.shape != coarse_labels.shape:
        raise ValueError("fine 与 coarse 标签形状不一致")
    matrix = np.zeros((len(fine_labels), num_fine + num_coarse), dtype=np.float32)
    rows = np.arange(len(fine_labels))
    matrix[rows, fine_labels] = 1.0
    matrix[rows, num_fine + coarse_labels] = 1.0
    return matrix

