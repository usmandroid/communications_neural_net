#%%


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import special as sp
from constants import k48, k96, k192, k288
from constants import w48, w96, w192, w288

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras import optimizers
from keras import initializers
from keras.callbacks import TensorBoard
import pickle
from time import time
from datetime import datetime
from scipy.special import logsumexp


def qfunc(arg):
    return 0.5 - 0.5 * sp.erf(arg / 1.414)


# In[55]:


QAM_64 = [[4, 12, 28, 20, 52, 60, 44, 36],
          [5, 13, 29, 21, 53, 61, 45, 37],
          [7, 15, 31, 23, 55, 63, 47, 39],
          [6, 14, 30, 22, 54, 62, 46, 38],
          [2, 10, 26, 18, 50, 58, 42, 34],
          [3, 11, 27, 19, 51, 59, 43, 35],
          [1, 9, 25, 17, 49, 57, 41, 33],
          [0, 8, 24, 16, 48, 56, 40, 32]]

QAM_16 = [[2, 6, 14, 10],
          [3, 7, 15, 11],
          [1, 5, 13, 9],
          [0, 4, 12, 8]]

QAM_4 = [[1, 3],
         [0, 2]]

BPSK_2 = [[0, 1]]

PSK_8 = [[0, 1, 5, 4, 6, 7, 3, 2]]
#
# QAM_64_b = [['000100', '001100', '011100', '010100', '110100', '111100', '101100', '100100'],
#             ['000101', '001101', '011101', '010101', '110101', '111101', '101101', '100101'],
#             ['000111', '001111', '011111', '010111', '110111', '111111', '101111', '100111'],
#             ['000110', '001110', '011110', '010110', '110110', '111110', '101110', '100110'],
#             ['000010', '001010', '011010', '010010', '110010', '111010', '101010', '100010'],
#             ['000011', '001011', '011011', '010011', '110011', '111011', '101011', '100011'],
#             ['000001', '001001', '011001', '010001', '110001', '111001', '101001', '100001'],
#             ['000000', '001000', '011000', '010000', '110000', '111000', '101000', '100000']]
#
# QAM_16_b = [['0000', '0100', '1100', '1000'],
#             ['0001', '0101', '1101', '1001'],
#             ['0011', '0111', '1111', '1011'],
#             ['0010', '0110', '1110', '1010']]
#
# QAM_4_b = [['01', '11'],
#            ['00', '10']]
#
# BPSK_2_b = [['0', '1']]

QAM_64_b = [['000100', '001100', '011100', '010100', '110100', '111100', '101100', '100100'],
            ['000101', '001101', '011101', '010101', '110101', '111101', '101101', '100101'],
            ['000111', '001111', '011111', '010111', '110111', '111111', '101111', '100111'],
            ['000110', '001110', '011110', '010110', '110110', '111110', '101110', '100110'],
            ['000010', '001010', '011010', '010010', '110010', '111010', '101010', '100010'],
            ['000011', '001011', '011011', '010011', '110011', '111011', '101011', '100011'],
            ['000001', '001001', '011001', '010001', '110001', '111001', '101001', '100001'],
            ['000000', '001000', '011000', '010000', '110000', '111000', '101000', '100000']]

QAM_16_b = [['0010', '0110', '1110', '1010'],
            ['0011', '0111', '1111', '1011'],
            ['0001', '0101', '1101', '1001'],
            ['0000', '0100', '1100', '1000']]

QAM_4_b = [['01', '11'],
           ['00', '10']]

BPSK_2_b = [['0', '1']]

PSK_8_b = [['000', '001', '101', '100', '110', '111', '011', '010']]


class Modulation_Map():
    def __init__(self, modulation):
        self.modulation = modulation
        if modulation == '4QAM':
            self.number_matrix, self.binary_matrix = QAM_4, QAM_4_b
        elif modulation == '16QAM':
            self.number_matrix, self.binary_matrix = QAM_16, QAM_16_b
        elif modulation == '64QAM':
            self.number_matrix, self.binary_matrix = QAM_64, QAM_64_b
        elif modulation == 'BPSK':
            self.number_matrix, self.binary_matrix = BPSK_2, BPSK_2_b
        elif modulation == '8PSK':
            self.number_matrix, self.binary_matrix = PSK_8, PSK_8_b
        self.create_num_to_bin_dictionary()
        self.create_bin_to_coordinate_dictionary()

    def create_num_to_bin_dictionary(self):
        number_matrix = self.number_matrix
        binary_matrix = self.binary_matrix
        n, d = len(number_matrix), len(number_matrix[0])
        num_to_bin = {}
        for i in range(n):
            for j in range(d):
                key = number_matrix[i][j]
                value = binary_matrix[i][j]
                num_to_bin[key] = value
        self.num_to_bin = num_to_bin

    def create_bin_to_coordinate_dictionary(self):
        number_matrix = self.number_matrix
        binary_matrix = self.binary_matrix
        n, d = len(number_matrix), len(number_matrix[0])
        bin_to_coordinate = {}
        for i in range(n):
            for j in range(d):
                key = binary_matrix[i][j]
                if self.modulation == '4QAM':
                    value = -1 + 2 * j, -1 + 2 * i
                elif self.modulation == '16QAM':
                    value = -3 + 2 * i, -3 + 2 * j
                elif self.modulation == '64QAM':
                    value = -7 + 2 * i, -7 + 2 * j
                elif self.modulation == 'BPSK':
                    value = -1 + 2 * j, i
                else:  # 8PSK
                    value = math.sin(j * math.pi / 4), math.cos(j * math.pi / 4)
                bin_to_coordinate[key] = value
                self.bin_to_coordinate = bin_to_coordinate




class Transmitter_Receiver():
    def __init__(self, num_bits_send, modulation, llr_calc='exact', scale_modulation=True):
        self.num_bits_send = num_bits_send
        self.modulation = modulation
        self.modulation_map = Modulation_Map(modulation)
        self.Scrambling = True
        self.Interleaving = True
        self.llr_calc = llr_calc
        self.scale_modulation = scale_modulation

        if modulation == '4QAM':
            self.M = 4;
            self.k = 2;
            self.NCBPS = 96
            self.binary_matrix = QAM_4_b
        elif modulation == '16QAM':
            self.M = 16
            self.k = 4;
            self.NCBPS = 192;
            self.binary_matrix = QAM_16_b
        elif modulation == '64QAM':
            self.M = 64;
            self.k = 6;
            self.NCBPS = 288;
            self.binary_matrix = QAM_64_b
        elif modulation == 'BPSK':
            self.M = 2;
            self.k = 1;
            self.NCBPS = 48;
            self.binary_matrix = BPSK_2_b
        else:  # 8PSK
            self.M = 8;
            self.k = 3;
            self.binary_matrix = PSK_8_b

        self.scaling_factor = self.get_scaling()

    def send_n_receive(self, snr, verbose=False):
        self.snr = snr
        N_0 = self.snr_to_N0()
        self.N_0 = N_0
        if verbose:
            print('Sending %d bits with snr = %fdB' % (self.num_bits_send, snr))
        self.generate_bits()
        self.serial_to_parallel()
        y = self.bit_stream_to_constellation()
        r = self.add_noise(y, N_0 / 2)
        self.r = r
        if verbose:
            plt.scatter(r[0, :], r[1, :])
            print('Finished Sending %d bits with snr = %fdB\n' % (self.num_bits_send, snr))

    def decode(self, verbose=False):
        if verbose:
            print('Decoding %d bits...' % (self.num_bits_send))
        r = self.r
        # print(r.shape)
        # print(r)
        n, d = np.shape(r)
        llr = np.zeros((self.k, d))
        decoded = np.zeros((self.k, d))
        for i in range(d):
            if self.llr_calc == 'approx':
                single_message_llr = self.r_to_llr_approx(r[:, i])
            if self.llr_calc == 'exact':
                single_message_llr = self.r_to_llr(r[:, i])
            # print(single_message_llr)
            llr[:, i] = np.flip(single_message_llr)
            decoded[:, i] = np.flip((1 - (single_message_llr / np.abs(single_message_llr))) / 2)

        bit_error = np.sum(np.abs(decoded - self.bit_stream))
        bit_error_rate = bit_error / self.num_bits_send
        self.bit_error = bit_error
        self.bit_error_rate = bit_error_rate
        self.llr = llr
        self.decoded = decoded
        if verbose:
            print('SNR level...%d' % (self.snr))
            print('Finished decoding %d bits...' % (self.num_bits_send))
            print('Number of Bit Errors: {}'.format(bit_error))
            print('Bit Error Rate:{}\n'.format(bit_error_rate))

    def get_scaling(self):
        scaling_dict = {1: 1, 2: np.sqrt(2), 4: np.sqrt(10), 6: np.sqrt(42)}
        k = self.k
        modulation = self.modulation
        if not self.scale_modulation:
            scaling_factor = 1
        else:
            scaling_factor = scaling_dict[k]
        return scaling_factor

    def r_to_llr(self, r):
        '''
    Returns: k array of llrs
    '''
        # print(r.shape)
        # print(r)
        k = self.k
        modulation_map = self.modulation_map
        llr = np.zeros(k)
        for i in np.arange(k):
            zero_sum, one_sum = 0., 0.
            num, den = 0., 0.
            num_values = [];
            den_values = []
            SCIPYLOGSUMEXP = False
            if SCIPYLOGSUMEXP:
                for key in modulation_map.bin_to_coordinate.keys():

                    r_ = modulation_map.bin_to_coordinate[key]
                    r_ = np.array([r_[0], r_[1]])
                    scale = 1 / self.scaling_factor
                    # custom_norm = np.abs(r[0] +1j*r[1])**2 +  np.abs(r_[0] +1j*r_[1])**2 -2*(r[0]*r_[0] + r[1]*r_[1])
                    exponent = -1.0 * (np.linalg.norm(scale * (r_ - r)) ** 2) / (2 * self.snr_to_N0_bpsk())
                    # exponent = -1 * custom_norm/(self.N_0)
                    if key[k - i - 1] == '0':
                        num_values.append(exponent)
                    else:
                        den_values.append(exponent)

                llr[i] = logsumexp(np.array(num_values)) - logsumexp(np.array(den_values))
                return llr
            else:

                for key in modulation_map.bin_to_coordinate.keys():
                    r_ = modulation_map.bin_to_coordinate[key]
                    r_ = np.array([r_[0], r_[1]])
                    scale = 1 / self.scaling_factor

                    exponent = -1.0 * (np.linalg.norm(scale * (r_ - r)) ** 2) / (2 * self.snr_to_N0_bpsk())
                    total = np.exp(exponent)
                    if key[k - i - 1] == '0':
                        zero_sum += total
                    else:
                        one_sum += total

                num = 0 if zero_sum == 0 else zero_sum
                den = 0 if one_sum == 0 else one_sum
                try:
                    num = np.log(zero_sum)
                    den = np.log(one_sum)
                except:
                    print(f"k = {k}")

            llr[i] = num - den
            llr[i] = 10 ** -5 if llr[i] == 0 else llr[i]
        return llr

    def r_to_llr_approx(self, r):
        '''
    Returns: k array of llrs
    '''
        k = self.k
        modulation_map = self.modulation_map
        llr = np.zeros(k)
        scale = 1 / self.scaling_factor
        for i in range(k):
            zero_sum, one_sum = [], []
            for key in modulation_map.bin_to_coordinate.keys():
                r_ = modulation_map.bin_to_coordinate[key]

                r_ = np.array([r_[0], r_[1]])
                # print(f"r_:  {r_.shape}")

                exponent = np.linalg.norm(scale * (r - r_)) ** 2
                total = exponent
                if key[k - i - 1] == '0':
                    zero_sum.append(total)
                else:
                    one_sum.append(total)
            llr[i] = (min(one_sum) - min(zero_sum)) * (1 / (2 * self.snr_to_N0_bpsk()))
            llr[i] = 10 ** -5 if llr[i] == 0 else llr[i]
        return llr

    def snr_to_N0(self):
        '''
    Returns: float N0
    '''
        k, M, snr = self.k, self.M, 10 ** (self.snr / 10)
        sum = 0
        for value in self.modulation_map.bin_to_coordinate.values():
            sum += value[0] ** 2 + value[1] ** 2
        e_avg = sum / M

        return e_avg / (k * snr)

    def snr_to_N0_bpsk(self):
        '''
    Returns: float N0 for BPSK
    '''
        k, M, snr = 1, 2, 10 ** (self.snr / 10)
        e_avg = 1
        return e_avg / (k * snr)

    def bi2de(self, binary):
        x = 0
        for n in range(0, len(binary)):
            x = (binary[n] * (2 ** n)) + x
        return x

    def de2bi(self, n, N):
        bseed = bin(n).replace("0b", "")
        fix = N - len(bseed)
        pad = np.zeros(fix)
        pad = pad.tolist()
        y = []
        for i in range(len(pad)):
            y = [int(pad[i])] + y
        for i in range(len(bseed)):
            y = [int(bseed[i])] + y
        return y

    def scrambler(self, seed=[1, 0, 1, 1, 1, 0, 1]):
        # initialize scrambler
        # scramble
        bits = self.bit_stream.ravel()
        bit_count = len(bits)
        scrambled_bits = np.zeros(bit_count)
        N = 7
        seed = self.bi2de(seed)
        bseed = self.de2bi(seed, N)
        x1 = bseed[0]
        x2 = bseed[1]
        x3 = bseed[2]
        x4 = bseed[3]
        x5 = bseed[4]
        x6 = bseed[5]
        x7 = bseed[6]

        for n in range(bit_count):
            x1t = x1
            x2t = x2
            x3t = x3
            x4t = x4
            x5t = x5
            x6t = x6
            x7t = x7
            var = int(x4t) ^ int(x7t)
            scrambled_bits[n] = int(var) ^ int(bits[n])
            x1 = var
            x2 = x1t
            x3 = x2t
            x4 = x3t
            x5 = x4t
            x6 = x5t
            x7 = x6t

        self.bit_stream = scrambled_bits.reshape((-1, 1))

    def generate_bits(self):
        '''
    Sets bit_stream to: num_bits_send, 1 array of bits
    '''
        # make num_bits_send a multiple of Nbpsc
        # e.g Nbpsc =  4 for 16-QAM
        # Nbits = self.num_bits_send + self.k - (self.num_bits_send % self.k)
        Nbits = self.num_bits_send
        # self.num_bits_send = Nbits
        bit_stream = np.around(np.random.rand(Nbits, 1))
        self.bit_stream = bit_stream

        # Scramble bits so there are not many repeated ones or zeros.

        if self.Scrambling:
            self.scrambler()

    def serial_to_parallel(self):
        '''
    Sets bit_stream to: k, n/k array of bits
    '''
        k = self.k
        c = self.bit_stream
        n, d = np.shape(c)

        if n % k == 0:
            copy = np.transpose(c.reshape((int(n / k), k)))
            self.bit_stream = copy
        else:
            copy = np.append(c, np.zeros((k - n % k, 1)))
            n_, = np.shape(copy)
            self.bit_stream = np.transpose(copy.reshape((int(n_ / k), k)))

    def bit_stream_to_constellation(self):
        '''
    Returns: 2,d array of 2-D coordinate points
    '''
        modulation = self.modulation
        modulation_map = self.modulation_map
        bit_stream = self.bit_stream
        n, d = np.shape(bit_stream)
        constellation = np.zeros((2, d))

        for j in range(d):
            stream = bit_stream[:, j]
            key = ''
            for bit in stream:
                key += str(int(bit))
            x1, y1 = modulation_map.bin_to_coordinate[key]
            constellation[0, j], constellation[1, j] = x1, y1
        return constellation

    def add_noise(self, y, var):
        n, d = np.shape(y)
        return y + math.sqrt(var) * np.random.randn(n, d)

# In[58]:


def theoretical_bit_error_rate(modulation, SNR_dB):
    snr = 10 ** (SNR_dB / 10)
    if modulation == 'BPSK':
        bit_error = qfunc((2 * snr) ** 0.5)
    elif modulation == '8PSK':
        M = 8
        bit_error = 2 / ((math.log2(M))) * qfunc(math.sin(math.pi / M) * (2 * snr * math.log2(M)) ** 0.5)
    elif modulation == '4QAM' or modulation == '16QAM' or modulation == '64QAM':
        M = 4 if modulation == '4QAM' else 16 if modulation == '16QAM' else 64
        bit_error = 4 / ((math.log2(M))) * qfunc((3 * snr * math.log2(M) / (M - 1)) ** 0.5)
    else:
        print('Must choose valid modulation')
        return 0
    return bit_error


NAME = ''
tensorboard = None


def init_tensorboard(modulation, train_SNR, llr_calc):
    global NAME
    global tensorboard
    NAME = "LLRnet-{}-snr{}-{}-{}".format(modulation, train_SNR, llr_calc, datetime.now().strftime("%Y%m%d_%H%M%S%U"))
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

class LLR_net():
    def __init__(self, modulation, training_size, test_size, train_snr=10,
                 test_snr=10, epoch_size=50, activation='tanh',
                 optimizer='adam', loss='mse', neuron_size=16,
                 layer_num=0, llr_calc='exact', input_dim=2, single_SNR=True):
        """


    :param modulation: 'BPSK' | '4QAM' | '16QAM' | '64QAM'
    :param training_size: 50000
    :param test_size: 10000
    :param train_snr: 10
    :param test_snr: 5
    :param epoch_size: 30
    :param activation: 'relu' | 'tanh'
    :param optimizer: 'adam' | 'sgd'
    :param loss: 'binary_crossentropy' | 'mse'
    :param neuron_size: K  =  8 |16 | 32 based on Modulation order
    :param layer_num: default = 0 (1st layer only), additional hidden to the 1st hidden
    :param llr_calc: 'exact' or 'approx' for training
    :param input_dim: 2 | 3
    """
        self.llr_calc = llr_calc
        self.activation = activation
        self.modulation = modulation
        self.neuron_size = 16 if self.modulation == '64QAM' else 8
        # self.neuron_size = neuron_size
        self.loss, self.optimizer = loss, optimizer
        self.train_snr, self.test_snr = train_snr, test_snr
        self.training_size, self.test_size = training_size, test_size
        self.model_history = {}
        self.input_dim = input_dim
        self.single_SNR = single_SNR
        init_tensorboard(self.modulation, self.train_snr, self.llr_calc)
        '''
    Model Definition: Sequential
    Model Paramters: The parameters can be specified in the declaration. Optimization of Parameters
                      are given in next sections
    Note:
    Last Layer is defined in the training system to account for different 'k' values
    '''
        model = Sequential()
        # Input Layer and First hidden layer
        model.add(Dense(self.neuron_size, input_dim=self.input_dim, activation=self.activation))
        # Extra Hidden Layers
        for i in range(layer_num):
            model.add(Dropout(0.2))
            model.add(Dense(self.neuron_size, activation=self.activation))
        self.model = model
        self.epoch_size = int(epoch_size)

    def train(self, wide_snrRange=[-10, 15], verbose=1):
        '''
    Generates Training Data by
    1. Defining a Transmitter/Receiver for the given number of training bits, modulation and snr
    2. Maps the bits to constellation with AWGN -> This is the Training Data, X
    3. Decodes using tradition decoding -> This is the Training Data, y
    4. Trains the Neural Network using the Training Data
    '''
        if self.single_SNR:

            self.training_system = Transmitter_Receiver(self.training_size, self.modulation, self.llr_calc)
            self.training_system.send_n_receive(self.train_snr)
            self.training_system.decode(verbose=1)
            if self.input_dim == 3:
                N_0_all = np.repeat([self.training_system.N_0],
                                    self.training_system.r.shape[1], axis=0).reshape(1, -1)
                self.X = np.transpose(np.concatenate((self.training_system.r, N_0_all), axis=0))
            if self.input_dim == 2:
                self.X = np.transpose(self.training_system.r)
            self.y = np.transpose(self.training_system.llr)
        else:
            self.wide_snrRange = wide_snrRange
            SNR_Range = self.wide_snrRange
            spacing = 1
            snr_list = np.linspace(SNR_Range[0], SNR_Range[1],
                                   int(np.abs(SNR_Range[0] - SNR_Range[1]) * 1 / spacing) + 1)
            train_size = self.training_size // len(snr_list)
            # train_size = 1000
            self.training_system = Transmitter_Receiver(train_size, self.modulation, self.llr_calc)

            k = self.training_system.k

            # print(f"X: {X.shape}"); print(f"y: {X.shape}")

            for snr_index, snr in enumerate(snr_list):
                self.training_system.send_n_receive(self.train_snr)
                self.training_system.decode(verbose=1)
                if snr_index == 0:
                    train_size = self.training_system.r.shape[1]
                    if self.input_dim == 3:
                        self.X = np.zeros((train_size * len(snr_list), 3), dtype=float)
                    if self.input_dim == 2:
                        self.X = np.zeros((train_size * len(snr_list), 2), dtype=float)
                    self.y = np.zeros((train_size * len(snr_list), self.training_system.k), dtype=float)
                if self.input_dim == 3:
                    N_0_all = np.repeat([self.training_system.N_0],
                                        self.training_system.r.shape[1], axis=0).reshape(1, -1)
                    X_i = np.transpose(np.concatenate((self.training_system.r, N_0_all), axis=0))
                if self.input_dim == 2:
                    X_i = np.transpose((self.training_system.r))
                y_i = np.transpose(self.training_system.llr)
                # print(f"r: {self.training_system.r.shape}"); print(f"X_i: {X_i.shape}"); print(f"y_i: {y_i.shape}")
                self.X[snr_index * train_size:(snr_index + 1) * train_size, :] = X_i
                self.y[snr_index * train_size:(snr_index + 1) * train_size, :] = y_i

        # Output Layer
        self.model.add(Dense(self.training_system.k, activation='linear'))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['mse'])
        print(f"X: {self.X.shape}");
        print(f"y: {self.y.shape}")
        self.model_history = self.model.fit(self.X, self.y, epochs=self.epoch_size, validation_split=0.3, batch_size=50,
                                            callbacks=[tensorboard], verbose=verbose)
        # Plot accuracy graph
        history = self.model_history
        self.plot_model_history(history)

    def test(self, verbose=True):
        '''
    Tests the model by:
    1. Defining a Transmitter/Receiver for the given number of testing bits, modulation and snr
    2. Maps the bits to constellation with AWGN -> This is the Training Data, X
    3. Decodes using tradition decoding -> This is the 'true' value, y
    '''
        self.test_system = Transmitter_Receiver(self.test_size, self.modulation, self.llr_calc)
        self.test_system.send_n_receive(self.test_snr)
        self.test_system.decode()
        N_0_all = np.repeat([self.test_system.N_0],
                            self.test_system.r.shape[1], axis=0).reshape(1, -1)
        if self.input_dim == 3:
            X = np.transpose(np.concatenate((self.test_system.r, N_0_all), axis=0))
        if self.input_dim == 2:
            X = np.transpose(self.test_system.r)
        self.predictions = self.model.predict(X)
        self.decode = np.transpose(0.5 * (-1 * self.predictions / np.abs(self.predictions) + 1))
        self.num_error = np.sum(np.abs(self.test_system.bit_stream - self.decode))
        self.b_error = self.num_error / self.test_system.num_bits_send
        self.conventional_error = self.test_system.bit_error / self.test_system.num_bits_send
        self.accuracy = 1 - np.average(np.abs(self.predictions.T - self.test_system.llr) / np.abs(self.test_system.llr))
        if verbose:
            print('Conventional Decoder bit error rate is %f' % (self.conventional_error))
            print('LLR Net bit error rate is %f' % (self.b_error))

    def plot_model_history(self, history=None):
        if history is None:
            history = self.model_history
        modulation = self.modulation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.plot(history.history['mse'])
        ax1.plot(history.history['val_mse'])
        ax1.set_title('model MSE')
        ax1.set_ylabel('MSE')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'test'], loc='upper left')
        # summarize history for loss
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'test'], loc='upper left')
        plotname = 'logs/{}/{}-{}_ModelHistory.png'.format(NAME, NAME, modulation)
        plt.savefig(plotname)

#

# %%
a = LLR_net('64QAM', 5000, 10000, train_snr = 0, test_snr = 5, activation = 'relu')
a.model.summary()




#%%
models = ['LLRnet-16QAM-snr20-exact-20230201_17200605/LLRnet-16QAM-snr20-exact-20230201_17200605_16QAM_model.pkl',
'LLRnet-4QAM-snr20-exact-20230201_17181505/LLRnet-4QAM-snr20-exact-20230201_17181505_4QAM_model.pkl',
'LLRnet-64QAM-snr20-exact-20230201_17214605/LLRnet-64QAM-snr20-exact-20230201_17214605_64QAM_model.pkl',
'LLRnet-BPSK-snr20-exact-20230201_17150205/LLRnet-BPSK-snr20-exact-20230201_17150205_BPSK_model.pkl']


# model_object_path = 'logs/' + models[0]
# with open(model_object_path, 'wb') as f:
#     pickle.dump(llrnet, f)
#
# with open(model_object_path, 'rb') as f:
#     a = pickle.load(f)

model_object_path = 'logs/LLRnet-64QAM-snr20-exact-20230201_17214605/LLRnet-64QAM-snr20-exact-20230201_17214605_64QAM_model.pkl'
#%%
with open(model_object_path, 'rb') as f:
    a =pickle.load(f)
a.model.summary()

# In[125]:

Channel = 'AWGN'
SNR_Range = [-10, 30]  # 0,31
spacing = 1
snr_list = np.linspace(SNR_Range[0], SNR_Range[1], int(np.abs(SNR_Range[0] - SNR_Range[1]) * 1 / spacing) + 1)
llr_calc = 'exact'
modulation_list = ['BPSK', '4QAM', '16QAM', '64QAM']  # 'BPSK','4QAM','16QAM','64QAM'
train_snr = 20
input_dim = 2  # 2 | 3
single_SNR = True
wide_snrRange = [0, 10]
training_size = 100000
test_size = 30000
epoch_size = 200
# modulation_list = [ 'BPSK','4QAM']
llr_net_accuracy = [[] for i in modulation_list]
# Track BER
llr_conventional = [[] for i in modulation_list]
conventional_error = [[] for i in modulation_list]
th_error = [[] for i in modulation_list]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 5))
axes = [ax1, ax2, ax3, ax4]
fig2, (ax11, ax22, ax33, ax44) = plt.subplots(1, 4, figsize=(25, 5))
axes2 = [ax11, ax22, ax33, ax44]

for n, modulation in enumerate(modulation_list):
    llrnet = LLR_net(modulation, training_size=training_size, test_size=10000, train_snr=train_snr,
                     activation='relu', llr_calc=llr_calc, input_dim=input_dim, single_SNR=single_SNR)
    llrnet.epoch_size = epoch_size
    llrnet.train(wide_snrRange=wide_snrRange)
    'logs/{}/{}_SNRvsBER.png'.format(NAME, NAME)
    model_object_path = 'logs/{}/{}_{}_model.pkl'.format(NAME, NAME, modulation)
    with open(model_object_path, 'wb') as f:
        pickle.dump(llrnet, f)

    with open(model_object_path, 'rb') as f:
        a = pickle.load(f)

    for snr in snr_list:
        # a = LLR_net(modulation, 10000, 20000, train_snr = snr, test_snr = snr, activation = 'relu')
        # a.train()
        a.test_snr = snr
        a.test()
        llr_net_accuracy[n].append(a.accuracy)
        llr_conventional[n].append(a.b_error)
        conventional_error[n].append(a.conventional_error)
        th_error[n].append(theoretical_bit_error_rate(modulation, snr))
    axes[n].plot(snr_list, llr_net_accuracy[n], linewidth=5, label='LLRnet Accuracy')
    axes[n].set_title(modulation)
    axes[n].set_ylim((0, 1))
    axes[n].set(xlabel='SNR (dB)', ylabel='Model Accuracy')
    axes2[n].semilogy(snr_list, llr_conventional[n], linewidth=5, label='Neural Net BER')
    axes2[n].semilogy(snr_list, conventional_error[n], label='Conventional BER')
    axes2[n].semilogy(snr_list, th_error[n], label='Theoretical BER')
    axes2[n].set_title(modulation)
    axes2[n].set(xlabel='snr (dB)', ylabel='Bit Error Rate')
ax1.legend()
ax11.legend()
plotname = 'logs/{}/{}_SNRvsMSE.png'.format(NAME, NAME)
fig.savefig(plotname)
plotname = 'logs/{}/{}_SNRvsBER.png'.format(NAME, NAME)
fig2.savefig(plotname)
