"""
This file contains model template and implementation for Forecaster.
All forecasting models should inherit from
ForecastModel, and override the _do_fit and _do_predict abstract methods
"""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class ForecastDataset(Dataset):
    """Data loader for time-series forecasting. Inherits torch.utils.data.Dataset

    Attributes
    ----------
    raw_df : pd.Dataframe
        A time-indexed dataframe of the sequence of counts, aggregated by
        the interval.
    horizon : pd.Timedelta
        How far into the future to predict (the y label).
    interval : pd.Timedelta
        The time interval to aggregate the original timeseries into.
    sequence_length : int
        The number of data points to use for prediction.
    X : torch.FloatTensor
        The (optionally transformed) tensors representing the training sequence
    y : torch.FloatTensor.
        The (optionally transformed) tensors expecting the expected value at the
        specified horizon.
    """

    def __init__(
        self,
        df,
        horizon=pd.Timedelta("1S"),
        interval=pd.Timedelta("1S"),
        sequence_length=5,
    ):
        self.horizon = horizon
        self.interval = interval
        self.sequence_length = sequence_length
        self.raw_df = df.resample(interval).sum()

        self.set_transformers((None, None))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1)]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i].reshape((1, -1))

    def get_y_timestamp(self, ind):
        """For a (seq,label) pair in the dataset, return the
        corresponding timestamp for the label.

        Parameters
        ----------
        ind : int
            Index into the dataset of the label in question.

        Returns
        -------
        pd.Timestamp : The timestamp of the corresponding label.
        """
        return self.raw_df.index[ind] + self.horizon

    def set_transformers(self, transformers):
        """Transform the X and y tensors according to the supplied transformers.
        Currently, both transformers are MinMaxScalers

        Parameters
        ----------
        transformers : (None, None) or Tuple of scikit.preprocessing data transformers
        """
        x_transformer, y_transformer = transformers
        shifted = self.raw_df.shift(freq=-self.horizon).reindex_like(self.raw_df).ffill()

        if x_transformer is None or y_transformer is None:
            self.X = torch.FloatTensor(self.raw_df.values)
            self.y = torch.FloatTensor(shifted.values)
            return

        self.X = torch.FloatTensor(x_transformer.transform(self.raw_df.values))
        self.y = torch.FloatTensor(y_transformer.transform(shifted.values))


class ForecastModel(ABC):
    """Interface for all the forecasting models"""

    def __init__(self, horizon, interval, sequence_length):
        self._x_transformer = None
        self._y_transformer = None

        self.horizon = horizon
        self.interval = interval
        self.sequence_length = sequence_length

    @property
    def name(self):
        return self.__class__.__name__

    def fit(self, train_seqs: ForecastDataset) -> None:
        """Fit the model with training sequences

        Parameters
        ----------
        train_seqs :
            List of training sequences and the expected output label
            in a certain horizon
        """

        # Make sure that the training data matches what the model expects
        assert self.horizon == train_seqs.horizon
        assert self.interval == train_seqs.interval
        assert self.sequence_length == train_seqs.sequence_length

        self._x_transformer, self._y_transformer = self._get_transformers(train_seqs.raw_df.values)

        transformed = copy.deepcopy(train_seqs)
        transformed.set_transformers((self._x_transformer, self._y_transformer))

        self._do_fit(transformed)

    @abstractmethod
    def _do_fit(self, trains_seqs: ForecastDataset) -> None:
        """Perform fitting.
        Should be overloaded by a specific model implementation.

        Parameters
        ----------
        train_seqs:
            List of training sequences and the expected output label in a
            certain horizon. Normalization would have been done if needed
        """
        raise NotImplementedError("Should be implemented by child classes")

    def predict(self, test_seq: np.ndarray) -> float:
        """Test a fitted model with a sequence.
        Parameters
        ----------
        test_seq:
            1D Test sequence

        Returns
        -------
        Predicted value at certain horizon
        """

        if self._x_transformer:
            test_seq = self._x_transformer.transform(test_seq)

        predict = self._do_predict(test_seq)
        if self._y_transformer:
            # Get the predicted scalar value back
            predict = self._y_transformer.inverse_transform(np.array([predict]).reshape(1, -1))[0][0]

        return predict

    @abstractmethod
    def _do_predict(self, test_seq: np.ndarray) -> float:
        """Perform prediction given input sequence.
        Should be overloaded by a specific model implementation.
        Parameters
        ----------
        test_seq
            1D Test sequence

        Returns
        -------
        Predicted value at certain horizon
        """
        raise NotImplementedError("Should be implemented by child classes")

    @abstractmethod
    def _get_transformers(self, data: np.ndarray) -> Tuple:
        """Get the data transformers
        Parameters
        ----------
        data :
            Training data

        Returns
        -------
        A tuple of x and y transformers
        """
        raise NotImplementedError("Each model should have its own transformers")

    def save(self, path):
        self._do_save(path)

    @abstractmethod
    def _do_save(self, path):
        raise NotImplementedError("Should be implemented by child classes")

    @staticmethod
    def load(path):
        raise NotImplementedError("Should be implemented by child classes")


class LSTM(nn.Module, ForecastModel):
    """A simple LSTM model serves as a template for ForecastModel"""

    def __init__(
        self,
        horizon: pd.Timedelta = pd.Timedelta(seconds=60),
        interval: pd.Timedelta = pd.Timedelta(seconds=10),
        sequence_length: int = 5,
        input_size: int = 1,
        hidden_layer_size: int = 100,
        num_hidden_layers: int = 1,
        output_size: int = 1,
        lr: float = 0.001,
        epochs: int = 10,
    ):
        """
        Parameters
        ----------
        input_size :
            Dimension of data point that is fed into the LSTM each time.
        hidden_layer_size :
            How many cells in one layer of the LSTM.
        num_hidden_layers :
            How many layers in the stacked LSTM.
        output_size :
            Dimension of prediction output.
        lr :
            Learning rate while fitting.
        epochs :
            Number of epochs for fitting.
        """
        nn.Module.__init__(self)
        ForecastModel.__init__(self, horizon, interval, sequence_length)

        self._hidden_layer_size = hidden_layer_size
        self._num_hidden_layers = num_hidden_layers

        self._lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_hidden_layers,
        )

        self._linear = nn.Linear(hidden_layer_size, output_size)

        self._hidden_cell = (
            torch.zeros(self._num_hidden_layers, 1, self._hidden_layer_size),
            torch.zeros(self._num_hidden_layers, 1, self._hidden_layer_size),
        )

        self._epochs = epochs
        self._lr = lr

    def forward(self, input_seq: torch.FloatTensor) -> float:
        """Forward propagation Implements nn.Module.forward().

        Parameters
        ----------
        input_seq : 1D FloatTensor

        Returns
        -------
        A single value prediction
        """
        lstm_out, self._hidden_cell = self._lstm(input_seq.view(len(input_seq), 1, -1), self._hidden_cell)
        predictions = self._linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    def _do_fit(self, train_seqs: Dataset) -> None:
        """
        Perform fitting.
        Should be overloaded by a specific model implementation.

        Parameters
        ----------
        train_seqs:
            List of training sequences and the expected output label in a
            certain horizon.
            Normalization has been done in ForecastModel.fit() with
            x and y transformers.
        """
        epochs = self._epochs
        lr = self._lr

        # Training specifics
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        logging.info(f"Training with {len(train_seqs)} samples, {epochs} epochs:")
        print(f"Training with {len(train_seqs)} samples, {epochs} epochs:")
        for i in range(epochs):
            # randomly shuffle training sequences
            arr = np.arange(len(train_seqs))
            np.random.shuffle(arr)
            for ind in arr:
                seq, labels = train_seqs[ind]
                optimizer.zero_grad()

                self._hidden_cell = (
                    torch.zeros(self._num_hidden_layers, 1, self._hidden_layer_size),
                    torch.zeros(self._num_hidden_layers, 1, self._hidden_layer_size),
                )
                seq = seq.view(-1)
                labels = labels.view(-1)

                y_pred = self(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i % 1 == 0:
                # logging.info(
                print(f"[LSTM FIT]epoch: {i + 1:3} loss: {single_loss.item():10.8f}")

        # logging.info(
        print(f"[LSTM FIT]epoch: {epochs:3} loss: {single_loss.item():10.10f}")

    def _do_predict(self, seq: np.ndarray) -> float:
        """Use LSTM to predict based on input sequence.
        Parameters
        ----------
        test_seq
            1D Test sequence.

        Returns
        -------
        Predicted value at certain horizon.
        """
        # To tensor
        seq = torch.FloatTensor(seq).view(-1)

        with torch.no_grad():
            self._hidden_cell = (
                torch.zeros(self._num_hidden_layers, 1, self._hidden_layer_size),
                torch.zeros(self._num_hidden_layers, 1, self._hidden_layer_size),
            )
            pred = self(seq)

        return pred.item()

    def _get_transformers(self, data: np.ndarray) -> Tuple:
        """
        Get the transformers. In the case of the LSTM, it uses the same
        MinMaxScaler for both X and Y.
        Parameters
        ----------
        data :
            Training data

        Returns
        -------
        A tuple of x and y transformers
        """
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)

        # Time-series data shares the same transformer
        return scaler, scaler

    def _do_save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        return torch.load(path)
