import matplotlib.pyplot as plt
import matplotlib


def plotBeforeAfter(before, after,ticks=[True,True]):
    """Helper function for image plot"""
    matplotlib.rcParams['figure.figsize'] = [20, 50]
    _, axs = plt.subplots(1,2)
    try:
        axs[0].imshow(before[::-1,:,:])
    except IndexError:
        axs[0].imshow(before[::-1,:],cmap="gray")
    axs[0].set_title("Before")
    axs[0].invert_yaxis()
    if not ticks[0]:
        axs[0].set_yticks([])
        axs[0].set_xticks([])
    else:
        axs[0].set_xlabel("x in millimeters")
        axs[0].set_ylabel("y in millimeters")
    try:
        axs[1].imshow(after[::-1,:,:])
    except IndexError:
        axs[1].imshow(after[::-1,:],cmap="gray")
    axs[1].set_title("After")
    axs[1].invert_yaxis()
    if not ticks[1]:
        axs[1].set_yticks([])
        axs[1].set_xticks([])
    else:
        axs[1].set_xlabel("x in millimeters")
        axs[1].set_ylabel("y in millimeters")