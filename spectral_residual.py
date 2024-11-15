import numpy as np

# Constant
EPS = 1e-8


class SpectralResidual:
    
    def __init__(self, mag_window, score_window, threshold):
        self.__mag_window = mag_window
        self.__score_window = score_window
        self.__threshold__ = threshold

    @staticmethod
    def average_filter(values, n):
        """
        Calculate the sliding window average for the give time series.
        Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
        :param values: list.
            a list of float numbers
        :param n: int, default 3.
            window size.
        :return res: list.
            a list of value after the average_filter process.
        """

        if n >= len(values):
            n = len(values)

        res = np.cumsum(values, dtype=float)
        res[n:] = res[n:] - res[:-n]
        res[n:] = res[n:] / n

        for i in range(1, n):
            res[i] /= (i + 1)

        return res

    def spectral_residual_transform(self, values):
        """
        This method transform a time series into spectral residual series
        :param values: list.
            a list of float values.
        :return: mag: list.
            a list of float values as the spectral residual values
        """

        trans = np.fft.fft(values)
        mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
        eps_index = np.where(mag <= EPS)[0]
        mag[eps_index] = EPS

        mag_log = np.log(mag)
        mag_log[eps_index] = 0

        avg_spectral = self.average_filter(mag_log, n=self.__mag_window)

        spectral = np.exp(mag_log - avg_spectral)

        trans.real = trans.real * spectral / mag
        trans.imag = trans.imag * spectral / mag
        trans.real[eps_index] = 0
        trans.imag[eps_index] = 0

        wave_r = np.fft.ifft(trans)
        saliencyMap = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
        return saliencyMap
    
    @staticmethod
    def predict_next(values):
        '''
        Predict the next value X_{n+1} by the average of the slope between X_n and X_{n-i} for i in [1, m];
        The predicted value of X_{n+1} is computed by X_{n-m+1}+m*avg_slope
        Mathematically, g = 1/m * sum_{i=1}^{m} g(x_n, x_{n-i}), x_{n+1} = x_{n-m+1} + g * m,
        where g(x_i,x_j) = (x_i - x_j) / (i - j)

        input: values: list
        output: float number
        '''

        if len(values)<=1:
            raise ValueError(f'Input list should contain at least 2 number')

        X_n = values[-1]
        n=len(values)
        slopes = [(values[n-1]-values[n-1-i])/i for i in range(1, n)]

        return values[0]+np.sum(slopes)/len(slopes)

    @staticmethod
    def extend_series(values, extend_num=5,look_ahead=5):
        '''
        Extend the input data by the predicted value
        Input: values: list
            extend_num: number of values to be added
            look_ahead: number of previous values used in prediction
        Output: list
        '''
        if look_ahead<2:
            raise ValueError(f'Number of previous values used in prediction must be at least 2')
        if look_ahead>len(values):
            look_ahead=len(values)

        # copy the 1st predicted points by k times, since it plays a decisive role.
        extension = [FodAnomalyDetection.predict_next(values[-look_ahead:])]* extend_num 

        return np.concatenate((values, extension), axis=0)

    def anomalyScore(self, saliencyMap, extend_num=5):
        '''
        compute the anomaly score: (x-moving_average(x))/moving_average(x),
            moving average: local average of the presiding q values of x
       
        '''

        avg_smap = self.average_filter(saliencyMap, self.__score_window)
        safeDivisors = np.clip(avg_smap, EPS, avg_smap.max())

        scores = (saliencyMap - avg_smap)/safeDivisors
        scores = scores[:(len(saliencyMap)-extend_num)]

        scores_clipped =np.clip(scores/10,0,1.0)

        return scores_clipped
