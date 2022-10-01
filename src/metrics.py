import torch


# F1-мера FROM BASELINE
class SoftDice:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, predictions, targets):
        numerator = torch.sum(2 * predictions * targets)
        denominator = torch.sum(predictions + targets)
        return (numerator + self.epsilon) / (denominator + self.epsilon)


# Метрика полноты BASELINE
class Recall:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, predictions, targets):
        numerator = torch.sum(predictions * targets)
        denominator = torch.sum(targets)

        return numerator / (denominator + self.epsilon)


# Метрика точности BASELINE
class Accuracy:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, predictions, targets):
        numerator = torch.sum(predictions * targets)
        denominator = torch.sum(predictions)

        return numerator / (denominator + self.epsilon)


def make_metrics():  # BASELINE
    soft_dice = SoftDice()
    recall = Recall()
    accuracy = Accuracy()

    def exp_dice(prediction, target):
        return soft_dice(torch.exp(prediction[:, 1:]), target[:, 1:])

    def exp_acc(prediction, target):
        return accuracy(torch.exp(prediction[:, 1:]), target[:, 1:])

    def exp_recall(prediction, target):
        return recall(torch.exp(prediction[:, 1:]), target[:, 1:])

    return [
        ('dice', exp_dice),
        ('accuracy', exp_acc),
        ('recall', exp_recall)
    ]


def make_criterion():  # BASELINE
    soft_dice = SoftDice()

    def exp_dice(prediction, target):
        return 1 - soft_dice(torch.exp(prediction[:, 1:]), target[:, 1:])

    return exp_dice


def make_bce_dice_criterion():
    soft_dice = SoftDice()

    def bce_dice(prediction, target):
        dice = 1 - soft_dice(torch.exp(prediction[:, 1:]), target[:, 1:])
        bce = torch.nn.functional.binary_cross_entropy(torch.exp(prediction[:, 1:]), target[:, 1:], reduction='mean')
        return bce + dice

    return bce_dice
