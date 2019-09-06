from math import pi
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import matlab.engine as engine
import matlab


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
eng = engine.start_matlab()

def phasecong2Matlab(inputTensor):
    inputTensor = inputTensor.cpu()
    inputTensor = inputTensor.squeeze(0).squeeze(0)
    inputTensor = inputTensor.to(torch.int) 
    return eng.phasecong2(matlab.double(inputTensor.tolist()))


def phasecong2(im):
    nscale = 4
    norient = 4
    minWaveLength = 6
    mult = 2
    sigmaOnf = 0.55
    dThetaOnSigma = 1.2
    k = 2.0
    epsilon = .0001

    thetaSigma  = pi / norient / dThetaOnSigma

    _, c, h, w = im.shape
    inputTensor = torch.zeros(c, h, w, 2).to(device)
    inputTensor[:, :, :, 0] = im.squeeze(0)
    imagefft = torch.fft(inputTensor, 2)

    zero = torch.zeros(h, w)

    E0 = cell(nscale, norient)

    estMeanE2n = []

    ifftFilterArray = cell(1, nscale)
    
    xRange = torch.Tensor([i-cols//2 for i in range(cols)])/cols

    yRange = torch.Tensor([i-rows//2 for i in range(rows)])/rows

    x = torch.stack([xRange for _ in range(len(xRange))])
    y = torch.stack([yRange for _ in range(len(yRange))])

    radius = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(-y, x)

    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius [0, 0] = 1


    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    lp = lowpassfilter([rows, cols], .45, 15)
    logGabor = [[]*nscale]
    for s in range(nscale):
        wavelength = minWaveLength * mult ** (s-1)
        fo = 1.00 / wavelength
        logGabor[s] = exp((-(log(radius/fo))**2) / (2 * math.log(sigmaOnf)**2))
        logGabor[s] = logGabor[s] * lp
        logGabor[s][0, 0] = 0

    spread = [[]*norient]
    
    for o in range(norient):
        angl = (0-1) * pi / norient
        ds = sintheta * cos(angl) - costheta * sin(angl)
        dc = costheta * cos(angl) - sintheta * sin(angl)
        dtheta = abs(atan2(ds, dc))
        spread[o] = exp((-dtheta ** 2) / (2* thetaSigma **2))

    # the main loop
    EnergyAll = torch.zeros(rows, cols)
    AnAll = torch.zeros(rows, cols)

def to4Dim(inputTensor):
    needs = 4 - inputTensor.ndim
    if needs >= 0:
        for _ in range(needs):
            inputTensor = inputTensor.unsqueeze(0)
    else:
        for _ in range(-needs):
            inputTensor = inputTensor.squeeze(0)
    return inputTensor.to(device)


def FeatureSIM(imageRef, imageDis):
    assert type(imageRef) == torch.Tensor, 'input image should be torch Tensor'
    assert type(imageDis) == torch.Tensor, 'input image should be torch Tensor'
    assert imageRef.shape == imageDis.shape, 'two input images should be of the same size'
    if imageRef.ndim == 2:
        print('dealing with gray images')
        print('not supporting gray images currently')
        raise ValueError
    if imageRef.ndim == 4:
        imageRef = imageRef.squeeze(0)
        imageDis = imageDis.squeeze(0)
    channels, rows, cols = imageRef.shape
    if imageRef.max() <= 1:
        imageRef = imageRef * 255
        imageDis = imageDis * 255

    if channels == 3:
        Y1 = 0.299 * imageRef[0] + 0.587 * imageRef[1] + 0.114 * imageRef[2]
        Y2 = 0.299 * imageDis[0] + 0.587 * imageDis[1] + 0.114 * imageDis[2]
        I1 = 0.596 * imageRef[0] - 0.274 * imageRef[1] - 0.322 * imageRef[2]
        I2 = 0.596 * imageDis[0] - 0.274 * imageDis[1] - 0.322 * imageDis[2]
        Q1 = 0.211 * imageRef[0] - 0.523 * imageRef[1] + 0.312 * imageRef[2]
        Q2 = 0.211 * imageDis[0] - 0.523 * imageDis[1] + 0.312 * imageDis[2]
    else:
        Y1 = imageRef.squeeze(0)
        Y2 = imageDis.squeeze(0)

    # Downsample the image
    minDimension = rows if cols > rows else cols
    F = max(1, round(minDimension / 256))
    aveKernel = torch.empty(1, 1, F, F).fill_(1/(F**2)).to(device)
    aveI1 = functional.conv2d(to4Dim(I1), aveKernel, padding=F//2)
    aveI2 = functional.conv2d(to4Dim(I2), aveKernel, padding=F//2)
    I1 = aveI1[:, :, 1:rows:F, 1:cols:F]
    I2 = aveI2[:, :, 1:rows:F, 1:cols:F]
    
    aveQ1 = functional.conv2d(to4Dim(Q1), aveKernel, padding=F//2)
    aveQ2 = functional.conv2d(to4Dim(Q2), aveKernel, padding=F//2)
    Q1 = aveQ1[:, :, 1:rows:F, 1:cols:F]
    Q2 = aveQ2[:, :, 1:rows:F, 1:cols:F]

    aveY1 = functional.conv2d(to4Dim(Y1), aveKernel, padding=F//2)
    aveY2 = functional.conv2d(to4Dim(Y2), aveKernel, padding=F//2)
    Y1 = aveY1[:, :, 1:rows:F, 1:cols:F]
    Y2 = aveY2[:, :, 1:rows:F, 1:cols:F]
    
    # calculate the phase congruency maps

    PC1 = phasecong2Matlab(Y1)
    PC2 = phasecong2Matlab(Y2)
    
    PC1 = torch.Tensor(PC1).to(torch.float).unsqueeze(0).unsqueeze(0)
    PC2 = torch.Tensor(PC2).to(torch.float).unsqueeze(0).unsqueeze(0)
    PC1 = PC1.to(device)
    PC2 = PC2.to(device)

    # calculate the gradient map

    dx = torch.Tensor([[[[3, 0, -3],
                       [10, 0, -10],
                       [3, 0, -3]]]])/16
    dy = torch.Tensor([[[[3, 10, 3],
                       [0, 0, 0],
                       [-3, -10, -3]]]])/16
    IxY1 = functional.conv2d(Y1, to4Dim(dx), padding = 1)
    IyY1 = functional.conv2d(Y1, to4Dim(dy), padding = 1)

    gradientMap1 = torch.sqrt(IxY1**2 + IyY1**2)
    
    IxY2 = functional.conv2d(Y2, to4Dim(dx), padding = 1)
    IyY2 = functional.conv2d(Y2, to4Dim(dy), padding = 1)

    gradientMap2 = torch.sqrt(IxY2**2 + IyY2**2)

    # calculate FSIM
    T1 = 0.85
    T2 = 160
    
    PCSimMatrix = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
    gradientSimMatrix = (2* gradientMap1 * gradientMap2 + T2) /(gradientMap1**2 + gradientMap2**2 + T2)
    PCm = torch.max(PC1, PC2)
    
    SimMatrix = gradientSimMatrix * PCSimMatrix * PCm
    FSIM = torch.sum(SimMatrix) / torch.sum(PCm)

    # Calculate the FSIMc
    T3 = 200
    T4 = 200
    ISimMatrix = (2 * I1 * I2 + T3) / (I1**2 + I2**2 + T3)
    QSimMatrix = (2 * Q1 * Q2 + T4) / (Q1**2 + Q2**2 + T4)
    Lambda = 0.03

    SimMatrixC = gradientSimMatrix * PCSimMatrix
    MatrixMiddle = (ISimMatrix * QSimMatrix) ** Lambda

    SimMatrixC = SimMatrixC * MatrixMiddle 
    SimMatrixC = SimMatrixC * PCm

    FSIMc = torch.sum(SimMatrixC) / torch.sum(PCm)

    return FSIMc.item()


if __name__ == "__main__":

    imageOne = Image.open('/home/resolution/Desktop/projTesting/originalData/2kOriginal/ParkJoy2_2560x1600_50_3.bmp')
    imageTwo = Image.open('/home/resolution/Desktop/projTesting/degradedData/noiseGaussian25/2k/ParkJoy2_2560x1600_50_3.bmp')
    print(FeatureSIM(ToTensor()(imageOne).unsqueeze(0), ToTensor()(imageTwo).unsqueeze(0)))
