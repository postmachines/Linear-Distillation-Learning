from collections import Counter


def get_few_shot_mnist(data_loader, shot=10):
    few_shot_dataset = []
    class_counter = Counter()
    for batch_i, (x, y) in enumerate(data_loader):
        if class_counter[y.item()] < shot:
            class_counter[y.item()] += 1
            few_shot_dataset.append((x, y))
        if all([x == shot for x in class_counter.values()]):
            break
    return few_shot_dataset
