import abc


class NLUTester(abc.ABC):
    @abc.abstractmethod
    def __init__(self, data):
        pass

    @abc.abstractmethod
    def run_tests(self):
        pass

    @abc.abstractmethod
    def compare_results(self):
        pass

    @abc.abstractmethod
    def save_results(self):
        pass

    @abc.abstractmethod
    def visualize_data(self):
        pass