import logging
import os

import numpy as np
import tensorflow as tf

from controller.mpc import MPC
from train_pinn import ManipulatorInformedNN
from utils.data import load_ref_trajectory, load_data
from utils.plotting import plot_input_sequence, plot_states, plot_absolute_error, animate
from utils.system import f

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logging.info("TensorFlow version: {}".format(tf.version.VERSION))
    logging.info("Eager execution: {}".format(tf.executing_eagerly()))           ## 여기까지 로그 출력

    # Paths
    resources_path = os.path.join('../resources')
    data_path = os.path.join(resources_path, 'data.npz')
    weights_path = os.path.join('../resources/weights')                          ## 여기까지 디렉토리 설정

    lb, ub, input_dim, output_dim, _, _, _, _ = load_data(data_path)             ## 여기까지 데이터 로드

    # Hyper parameter
    N_l = 4
    N_n = 64
    layers = [input_dim, *N_l * [N_n], output_dim]                               ## 여기까지 네트워크 설정

    logging.info('MPC parameters:')
    H = 5
    logging.info(f'\tH:\t{H}')                                                   ## 여기까지 MPC horizon 설정
    u_ub = np.array([0.5, 0.5])                                                     
    u_lb = - u_ub                                                                ## 여기까지 MPC input bound 설정                                 

    X_ref, T_ref = load_ref_trajectory(resources_path)                           ## 여기까지 로봇궤적ref 로드

    x0 = X_ref[0]
    T_ref = T_ref[:-H, 0]                                                        ## 여기까지 초기값 설정

    tau = T_ref[1] - T_ref[0]  
    logging.info(f'\ttau:\t{tau}')                                               ## 여기까지 sampling time 설정

    # Initialization
    pinn = ManipulatorInformedNN(layers, lb, ub)                                 ## NN는 Phys를 어케암???????????????????????
    # Load pretrained weights
    pinn.load_weights(weights_path)

    controller = MPC(f, pinn.model, u_ub=u_ub, u_lb=u_lb,
                     t_sample=tau, H=H,
                     Q=tf.linalg.tensor_diag(tf.constant([1, 1, 0, 0], dtype=tf.float64)),
                     R=1e-6 * tf.eye(2, dtype=tf.float64))                       ## system dynamics, PINN, input bound, sampling time, horizon, 마지막 두 개 뭐임???????????

    # Testing self loop prediction
    H_sl = 20                                                                    ## self loop horizon 설정

    # Generate self loop prediction input sequence
    U1_sl = 0.5 * np.sin(np.linspace(0, 2 * np.pi, H_sl))
    U2_sl = - U1_sl
    U_sl = np.hstack((U1_sl[:, np.newaxis], U2_sl[:, np.newaxis]))

    # Initial state
    x0_sl = np.zeros(4)

    # Simulate plant system
    X_ref_sl = controller.sim_open_loop_plant(x0_sl, U_sl,
                                              t_sample=tau,
                                              H=H_sl)

    # Simulate PINN system
    X_sl = controller.sim_open_loop(x0_sl, U_sl,
                                    t_sample=tau,
                                    H=H_sl)

    T_sl = np.arange(0., H_sl * tau + tau, tau)
    plot_input_sequence(T_sl, U=np.vstack((U_sl, U_sl[-1:, :])))
    plot_states(T_sl, X_ref_sl, X_sl)
    plot_absolute_error(T_sl, X_ref_sl, X_sl)

    # Testing closed loop
    X_mpc, U_mpc, X_pred = controller.sim(x0, X_ref, T_ref)

    plot_input_sequence(T_ref, U_mpc)
    plot_states(T_ref, X_ref[:-H], Z_mpc=X_mpc)
    plot_absolute_error(T_ref, X_ref[:-H], Z_mpc=X_mpc)
    animate(X_ref[:-H], [X_mpc], ['MPC'], fps=1 / tau)