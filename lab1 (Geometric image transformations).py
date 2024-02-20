import matplotlib.pyplot as plt
import scipy.misc as misc
import numpy as np


def affine_nn(Is, A, b, Hd, Wd):
    Hs, Ws, _ = Is.shape
    Id = np.zeros((Hd, Wd, Is.shape[2] if len(Is.shape) > 2 else 1), dtype=Is.dtype)

    # print("TEST:", A)

    for r_d in range(Hd):
        for c_d in range(Wd):
            p_d = np.array([r_d, c_d])
            
            # print("TEST:", b)
            # print("TEST:", p_d)
            
            p_s = np.dot(A, p_d) + b

            r_s, c_s = int(round(p_s[0])), int(round(p_s[1]))

            if 0 <= r_s < Hs and 0 <= c_s < Ws:
                Id[r_d, c_d] = Is[r_s, c_s]

    return Id


def affine_bilin(Is, A, b, Hd, Wd):
    Hs, Ws = Is.shape[:2]
    Id = np.zeros((Hd, Wd, Is.shape[2] if len(
        Is.shape) > 2 else 1), dtype=Is.dtype)

    for r_d in range(Hd):
        for c_d in range(Wd):
            p = np.dot(A, np.array([r_d, c_d])) + b
            r_s, c_s = p.astype(int)

            if 0 <= r_s < Hs - 1 and 0 <= c_s < Ws - 1:
                dr = p[0] - r_s
                dc = p[1] - c_s
                Id[r_d, c_d] = (1 - dr) * ((1 - dc) * Is[r_s, c_s] + dc * Is[r_s, c_s + 1]) + \
                    dr * ((1 - dc) * Is[r_s + 1, c_s] +
                          dc * Is[r_s + 1, c_s + 1])

    return Id

def recover_affine_diamond(Hs, Ws, Hd, Wd):
    Qs = np.array([[Ws / 2, Hs / 2], [Ws / 2, 0], [Hs / 2, Hs]])
    Qd = np.array([[Wd / 2, 0],[Wd / 2, 0], [Hd / 2, Hd]])

    A = np.zeros((6, 6))
    b = np.zeros(6)

    for i in range(3):
        xi, yi = Qs[i]
        xi_p, yi_p = Qd[i]
        A[i * 2, :] = [xi, yi, 0, 0, 1, 0]
        A[i * 2 + 1, :] = [0, 0, xi, yi, 0, 1]
        b[i * 2] = xi_p
        b[i * 2 + 1] = yi_p

    x = np.linalg.solve(A, b)

    A_affine = np.zeros((2, 3))
    A_affine[0, 0] = x[0]
    A_affine[0, 1] = x[1]
    A_affine[0, 2] = x[2]
    A_affine[1, 0] = x[3]
    A_affine[1, 1] = x[4]
    A_affine[1, 2] = x[5]

    return A_affine, b

def recover_projective(Qs, Qd):
    # Stvorite homogene koordinate
    Qs_homo = np.vstack((Qs, np.ones(Qs.shape[1])))
    Qd_homo = np.vstack((Qd, np.ones(Qd.shape[1])))

    # Izračunajte matricu transformacije pomoću SVD
    _, _, V = np.linalg.svd(Qs_homo)
    H = V[-1, :].reshape(3, 3)

    # Normalizacija matrice transformacije
    H = H / H[2, 2]

    return H




Is = misc.face()
Is = np.asarray(Is)

Hd, Wd = 200, 200
A, b = recover_affine_diamond(Is.shape[0], Is.shape[1], Hd, Wd)
print(A)
# A = 0.25 * np.eye(2) + np.random.normal(size=(2, 2))

Id1 = affine_nn(Is, A[:,:2], b, Hd, Wd)
Id2 = affine_bilin(Is, A[:,:2], b, Hd, Wd)

fig = plt.figure()
if len(Is.shape) == 2:
    plt.gray()
for i, im in enumerate([Is, Id1, Id2]):
    fig.add_subplot(1, 3, i+1)
    plt.imshow(im.astype(int))
plt.show()
