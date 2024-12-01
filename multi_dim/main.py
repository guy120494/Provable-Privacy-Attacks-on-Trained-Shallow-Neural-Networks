import statistics

import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform_direction
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

from multi_dim.neuralNetwork import NeuralNetwork

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

# Define the number of epochs
N_EPOCHS = 100


def find_marginal_points(dataloader, model):
    """
    Finding all points that lie on the margin. We allow a 10% slack
    :param dataloader: A PyTorch DataLoader object that contains the points
    :param model: The model where we want to find the marginal points
    :return: List of all points x in dataloader such that model(x) lie on the margin up to 10% slack
    """
    marginal_points = []
    margin_value = float('inf')
    with torch.no_grad():
        for training_points, training_labels in dataloader:
            training_points = training_points.to(device)
            training_labels = training_labels.to(device)
            training_preds = model(training_points).cpu().detach().numpy().squeeze()
            training_labels = training_labels.cpu().detach().numpy().squeeze()

            # Finding the margin's value, which is the smallest positive value of y_i * x_i
            for training_pred, training_label in zip(training_preds, training_labels):
                if 0 < training_pred * training_label < margin_value:
                    margin_value = abs(training_pred * training_label)

        for training_points, training_labels in dataloader:
            training_points = training_points.to(device)
            training_labels = training_labels.to(device)
            training_preds = model(training_points).cpu().detach().numpy().squeeze()
            training_labels = training_labels.cpu().detach().numpy().squeeze()

            # Finding all points (x,y) for which margin < y * phi(x) < 1.1 * margin - giving a 10% slack
            for training_point, training_pred, training_label in zip(training_points, training_preds, training_labels):
                if 0 < training_pred * training_label < 1.1 * margin_value:
                    marginal_points.append(training_point.cpu().detach().numpy())

    return np.array(marginal_points), margin_value


def create_set_of_points_on_sphere(number_of_points=100, points_dim=2):
    """
    Creating a labeled set of points on a sphere.
    :param number_of_points: number of points to create
    :param points_dim: the dimension of each point
    :return: A labeled set of points on the sphere. The labels divide the sphere into 2^points_dim orthants
    where each orthant has the same label
    """
    total_x = []
    total_y = []
    for i in range(0, number_of_points, 10):
        x = uniform_direction.rvs(dim=points_dim, size=10)
        x *= np.sqrt(points_dim)
        y = np.logical_xor.reduce((np.sign(x) > 0), axis=1).astype(float) * 2 - 1
        total_x.append(x)
        total_y.append(y)

    total_x = np.concatenate(total_x, axis=0).astype(np.float32)
    total_y = np.expand_dims(np.concatenate(total_y, axis=0), -1).astype(np.float32)
    total_x = torch.from_numpy(total_x)
    total_y = torch.from_numpy(total_y)
    dataset = TensorDataset(total_x, total_y)
    dataloader = DataLoader(dataset, batch_size=number_of_points, shuffle=True)
    return dataloader


def create_mixture_of_gaussian_points(number_of_points=100, points_dim=2):
    total_x = []
    total_y = []
    for i in range(0, number_of_points, 10):
        k = 2 * np.random.randint(0, 2) - 1
        mean = [k] + [0 for _ in range(points_dim - 1)]
        cov = np.eye(points_dim)
        x = np.random.multivariate_normal(mean, cov, 10)
        y = np.logical_xor.reduce((np.sign(x) > 0), axis=1).astype(float) * 2 - 1
        total_x.append(x)
        total_y.append(y)

    total_x = np.concatenate(total_x, axis=0).astype(np.float32)
    total_y = np.expand_dims(np.concatenate(total_y, axis=0), -1).astype(np.float32)
    total_x = torch.from_numpy(total_x)
    total_y = torch.from_numpy(total_y)
    dataset = TensorDataset(total_x, total_y)
    dataloader = DataLoader(dataset, batch_size=number_of_points, shuffle=True)
    return dataloader


def get_number_of_test_points_greater_than_margin(test_loader, model, margin_value):
    """
    computing the number of test points greater than or equal to margin_value
    :param test_loader: A data loader object that contains the points
    :param model: The model that we want to compute the number of test points greater than margin_value
    :param margin_value: the value of the margin value
    :return: The number of points x in test_loader that model(x) is greater than or equal to margin_value
    """
    all_points = []
    all_labels = []
    all_preds = []
    bad_points = []
    with torch.no_grad():
        # Loop over batches using DataLoader
        for id_batch, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch_pred = model(x_batch)

            x_batch = x_batch.cpu().detach().numpy()
            y_batch = y_batch.cpu().detach().numpy().squeeze()
            y_batch_pred = y_batch_pred.cpu().detach().numpy().squeeze()
            for point, pred, label in zip(x_batch, y_batch_pred, y_batch):
                if pred * label >= margin_value:
                    bad_points.append([point, pred, label])
            all_points.append(x_batch)
            all_labels.append(y_batch)
            all_preds.append(y_batch_pred)

        return len(bad_points)


def get_trained_model(input_dim, dataloader):
    """
    Get a trained neural network model, using BCE loss
    :param input_dim: The dimension of the input
    :param dataloader: A dataloader object that contains the training points
    :return: A trained neural network model
    """
    small_constant_initializer = 1e-3
    model = NeuralNetwork(input_dim=input_dim, hidden_layer_dim=10000)
    with torch.no_grad():
        for layer in model.parameters():
            layer.data = layer.data * small_constant_initializer
    model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.7)
    # Get the dataset size for printing (it is equal to N_SAMPLES)
    dataset_size = len(dataloader.dataset)
    # Loop over epochs
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        # Loop over batches in an epoch using DataLoader
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch_pred = model(x_batch)
            loss = loss_fn(y_batch_pred, (y_batch + 1) / 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 100 batches, print the loss for this batch
            # as well as the number of examples processed so far
            if id_batch % 100 == 0:
                loss, current = loss.item(), (id_batch + 1) * len(x_batch)
                print(f"loss: {loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")
                print(f"learning rate: {scheduler.optimizer.param_groups[0]['lr']}")

        # Step the scheduler after each epoch
        scheduler.step()

    return model


def plot_graph_of_percentage_of_marginal_point_as_a_function_of_dimension(input_dims, train_size,
                                                                          distribution='sphere'):
    """
    Plotting the graph of the percentage of marginal points (up to a 10% slack) as a function of the input's dimension
    :param input_dims: the dimension of the input data
    :param train_size: the size of the training data
    """
    number_of_marginal_points = []
    for input_dim in input_dims:
        print(f'INPUT DIM: {input_dim}')
        if distribution == 'sphere':
            dataloader = create_set_of_points_on_sphere(number_of_points=train_size, points_dim=input_dim)
        else:
            dataloader = create_mixture_of_gaussian_points(number_of_points=train_size, points_dim=input_dim)
        model = get_trained_model(input_dim, dataloader)

        marginal_points, _ = find_marginal_points(dataloader, model)
        number_of_marginal_points.append(marginal_points.shape[0] / train_size)

    plt.rcParams.update({'font.size': 34})
    plt.figure(figsize=(12, 10))
    plt.title("Ratio of Marginal Points", fontsize=40, wrap=True)
    plt.xlabel("Input's dimension", fontsize=38, wrap=True)
    plt.ylabel("Ratio of marginal points", fontsize=38, wrap=True)
    plt.plot(input_dims, number_of_marginal_points)
    plt.ylim(ymin=0)
    # plt.savefig(rf'C:\Users\admin\OneDrive\Desktop\privacyIssuesOneDim\ratio_of_marginal_points.eps', format='eps', dpi=1200)
    plt.show()


def plot_graph_of_average_percentage_of_marginal_point_as_a_function_of_dimension(input_dims, train_size,
                                                                                  distribution='sphere',
                                                                                  number_of_trials=10):
    """
    Plotting the graph of the percentage of marginal points (up to a 10% slack) as a function of the input's dimension
    :param number_of_trials: number of trials for each dimension
    :param input_dims: the dimension of the input data
    :param train_size: the size of the training data
    """
    average_number_of_marginal_points = []
    for input_dim in input_dims:
        print(f'INPUT DIM: {input_dim}')
        number_of_marginal_points = []
        for _ in range(number_of_trials):
            if distribution == 'sphere':
                dataloader = create_set_of_points_on_sphere(number_of_points=train_size, points_dim=input_dim)
            else:
                dataloader = create_mixture_of_gaussian_points(number_of_points=train_size, points_dim=input_dim)
            model = get_trained_model(input_dim, dataloader)
            marginal_points, _ = find_marginal_points(dataloader, model)
            number_of_marginal_points.append(marginal_points.shape[0] / train_size)
        average_number_of_marginal_points.append(statistics.fmean(number_of_marginal_points))

    plt.rcParams.update({'font.size': 34})
    plt.figure(figsize=(12, 10))
    plt.title("Average Ratio of Marginal Points", fontsize=40, wrap=True)
    plt.xlabel("Input's dimension", fontsize=38, wrap=True)
    plt.ylabel("Average Ratio of marginal points", fontsize=38, wrap=True)
    plt.plot(input_dims, average_number_of_marginal_points)
    plt.ylim(ymin=0)
    plt.savefig(
        rf'C:\Users\admin\OneDrive\Desktop\privacyIssuesOneDim\average_ratio_of_marginal_points_{distribution}.eps',
        format='eps', dpi=1200)
    plt.show()


def plot_graph_of_bad_test_points_as_a_function_of_dimension(input_dims, train_size, test_size, distribution='sphere'):
    """
    Plotting a graph of the percentage of test points that lie on or above the margin as a
    function of the input's dimension. For each dimension, a new model is trained
    :param input_dims: the input's dimension
    :param train_size: the size of the training data
    :param test_size: the size of the test data
    """
    ratio_of_bad_points = []
    for input_dim in input_dims:
        print(f'INPUT DIM: {input_dim}')
        if distribution == 'sphere':
            dataloader = create_set_of_points_on_sphere(number_of_points=train_size, points_dim=input_dim)
        else:
            dataloader = create_set_of_points_on_sphere(number_of_points=train_size, points_dim=input_dim)
        model = get_trained_model(input_dim, dataloader)

        marginal_points, margin_value = find_marginal_points(dataloader, model)
        print(f"margin value: {margin_value}")
        print(f"TEST")
        if distribution == 'sphere':
            test_set = create_set_of_points_on_sphere(number_of_points=test_size, points_dim=input_dim)
        else:
            test_set = create_mixture_of_gaussian_points(number_of_points=test_size, points_dim=input_dim)
        ratio_of_bad_points.append(
            get_number_of_test_points_greater_than_margin(test_set, model, margin_value) / test_size)

    plt.rcParams.update({'font.size': 34})
    plt.figure(figsize=(12, 10))
    plt.title("Ratio of test points with value greater or equal to margin", fontsize=40, wrap=True)
    plt.xlabel("Input's dimension", fontsize=38, wrap=True)
    plt.ylabel("Ratio", fontsize=38, wrap=True)
    plt.plot(input_dims, ratio_of_bad_points)
    # plt.savefig(rf'C:\Users\admin\OneDrive\Desktop\privacyIssuesOneDim\average_ratio_of_bad_points_up_to_dim_{input_dims[-1]}.eps', format='eps', dpi=1200)
    plt.show()


def plot_graph_average_of_bad_test_points_as_a_function_of_dimension(input_dims, train_size, test_size,
                                                                     distribution='sphere',
                                                                     number_of_trials=10):
    """
    Plotting a graph of the percentage of test points that lie on or above the margin as a
    function of the input's dimension. For each dimension, a new model is trained
    :param input_dims: the input's dimension
    :param train_size: the size of the training data
    :param test_size: the size of the test data
    :param number_of_trials: number of trials
    """
    average_ratio_of_bad_points = []
    for input_dim in input_dims:
        print(f'INPUT DIM: {input_dim}')
        ratio_of_bad_points = []
        for _ in range(number_of_trials):
            if distribution == 'sphere':
                dataloader = create_set_of_points_on_sphere(number_of_points=train_size, points_dim=input_dim)
            else:
                dataloader = create_mixture_of_gaussian_points(number_of_points=train_size, points_dim=input_dim)
            model = get_trained_model(input_dim, dataloader)

            marginal_points, margin_value = find_marginal_points(dataloader, model)
            print(f"margin value: {margin_value}")
            print(f"TEST")
            if distribution == 'sphere':
                test_set = create_set_of_points_on_sphere(number_of_points=test_size, points_dim=input_dim)
            else:
                test_set = create_mixture_of_gaussian_points(number_of_points=test_size, points_dim=input_dim)
            ratio_of_bad_points.append(
                get_number_of_test_points_greater_than_margin(test_set, model, margin_value) / test_size)
        average_ratio_of_bad_points.append(statistics.fmean(ratio_of_bad_points))

    plt.rcParams.update({'font.size': 34})
    plt.figure(figsize=(12, 10))
    plt.title("Average ratio of test points with value greater or equal to margin", fontsize=40, wrap=True)
    plt.xlabel("Input's dimension", fontsize=38, wrap=True)
    plt.ylabel("Average ratio", fontsize=38, wrap=True)
    plt.plot(input_dims, average_ratio_of_bad_points)
    plt.savefig(
        rf'C:\Users\admin\OneDrive\Desktop\privacyIssuesOneDim\average_ratio_of_bad_points_up_to_dim_{input_dims[-1]}_{distribution}.eps',
        format='eps', dpi=1200)
    plt.show()


if __name__ == '__main__':
    # plot_graph_of_percentage_of_marginal_point_as_a_function_of_dimension(input_dims=[i for i in range(100, 1200, 50)],
    #                                                                       train_size=20)

    # plot_graph_of_average_percentage_of_marginal_point_as_a_function_of_dimension(input_dims=[i for i in range(100, 1200, 50)],
    #                                                                       train_size=20, number_of_trials=50)

    # plot_graph_average_of_bad_test_points_as_a_function_of_dimension(input_dims=[i for i in range(1, 100, 1)],
    #                                                                  train_size=20,
    #                                                                  test_size=5000, number_of_trials=50)
    # long_range = [i for i in range(1, 100, 1)]
    # long_range.extend([i for i in range(100, 600, 10)])
    # plot_graph_average_of_bad_test_points_as_a_function_of_dimension(input_dims=long_range,
    #                                                                  train_size=20,
    #                                                                  test_size=5000,
    #                                                                  number_of_trials=50)

    # plot_graph_of_average_percentage_of_marginal_point_as_a_function_of_dimension(
    #     input_dims=[i for i in range(100, 1200, 50)],
    #     train_size=20,
    #     distribution='gaussian',
    #     number_of_trials=50)

    # plot_graph_average_of_bad_test_points_as_a_function_of_dimension(input_dims=[i for i in range(1, 100, 1)],
    #                                                                  train_size=20,
    #                                                                  test_size=5000,
    #                                                                  number_of_trials=50,
    #                                                                  distribution='gaussian')
    long_range = [i for i in range(1, 100, 1)]
    long_range.extend([i for i in range(100, 600, 10)])
    plot_graph_average_of_bad_test_points_as_a_function_of_dimension(input_dims=long_range,
                                                                     train_size=20,
                                                                     test_size=5000,
                                                                     number_of_trials=50,
                                                                     distribution='gaussian')