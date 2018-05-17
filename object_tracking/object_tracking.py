"""OpenCV object tracking"""
#pylint: disable=global-statement
import argparse
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

__version_info__ = (1, 0, 2)
__version__ = '.'.join([str(n) for n in __version_info__])

LICENSE_MSG = """\
OpenCV object tracking
MIT License
Copyright (c) 2018 Paul Freeman"""

# global constants
ESCAPE_KEY = 27
PLOT_TITLE = 'Tracked Object Motion'
PLOT_XLABEL = 'Time [seconds]'
PLOT_YLABEL = 'Distance [metres]'

def tracking():
    """Perform object tracking"""
    print(LICENSE_MSG)
    args = parse_args()
    try:
        write_mode = 'wb' if args.force else 'xb'
        with open(args.output_file, write_mode) as output_file:
            video = open_video(args.video_file)
            read_frame(video)
            print(SET_SCALE_MSG)
            press_enter()
            print('<scale window displayed>')
            distance = get_scale_distance()
            print('\nThe line drawn has a distance of {:.1f} pixels.'.format(distance))
            measure = float(input('Tell me how many metres this should represent. > '))
            scale = distance / measure
            print(ROI_BOX_MSG)
            press_enter()
            print('<object tracking window displayed>')
            bbox = select_bounding_box()
            if args.algorithm == 'KCF':
                tracker = cv2.TrackerKCF_create()
            elif args.algorithm == 'MIL':
                tracker = cv2.TrackerMIL_create()
            elif args.algorithm == 'Median-Flow':
                tracker = cv2.TrackerMedianFlow_create()
            else:
                raise ValueError('Unknown algorithm type')
            if args.suppress_live_plot:
                print(TRACKING_MSG)
            else:
                print(TRACKING_MSG_W_PLOT)
            press_enter()
            print('<object tracking window displayed>')
            points = track_video(video, tracker, bbox, scale, args.suppress_live_plot, args.algorithm)
            np.save(output_file, np.asarray(points))
            print(LAST_PLOT_MSG)
            press_enter()
            plt.cla()
            plt.scatter(points.T[0], points.T[1])
            plt.title(PLOT_TITLE)
            plt.xlabel(PLOT_XLABEL)
            plt.ylabel(PLOT_YLABEL)
            plt.show()
    except FileExistsError:
        print('This directory already contains a file named: {}'.format(args.output_file))
        print('Please move, rename, or delete this file and try again.')


def read_frame(video):
    """Read a frame into the global variable"""
    global FRAME, COPY
    frame_read_success, FRAME = video.read()
    if not frame_read_success:
        raise RuntimeError('Could not read specified video file')
    COPY = FRAME.copy()


def get_scale_distance():
    """Calculates the scale from a drawn line on the video"""
    cv2.namedWindow('scale')
    cv2.setMouseCallback('scale', draw_line)
    while True:
        cv2.imshow('scale', COPY)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('\n') or k == ord('\r'):
            break
    cv2.destroyWindow('scale')
    return sqrt((X_VAL_2-X_VAL_1)**2 + (Y_VAL_2-Y_VAL_1)**2)


def select_bounding_box():
    """Mark a bounding box to be tracked"""
    try:
        return cv2.selectROI(FRAME, False)
    finally:
        cv2.destroyAllWindows()


def track_video(video, tracker, bbox, scale, suppress_live_plot, algorithm):
    """Track a video"""
    fps = video.get(cv2.CAP_PROP_FPS)
    if not tracker.init(FRAME, bbox):
        raise RuntimeError('Could not initialize video file')
    origin = ((2.0 * bbox[0] + bbox[2]) / 2.0,
              (2.0 * bbox[1] + bbox[3]) / 2.0)
    frame_number = 0
    time_points = [frame_number / fps]
    dist_points = [0]
    if not suppress_live_plot:
        plt.ion()
        plt.scatter(time_points, dist_points)
        plt.title(PLOT_TITLE)
        plt.xlabel(PLOT_XLABEL)
        plt.ylabel(PLOT_YLABEL)
        plt.show()
    while True:
        try:
            read_frame(video)
            frame_number += 1
        except RuntimeError:
            break
        tracking_success, bbox = tracker.update(FRAME)
        if not tracking_success:
            print(TRACKING_FAIL_MSG)
            if algorithm == 'KCF':
                tracker = cv2.TrackerKCF_create()
            elif algorithm == 'MIL':
                tracker = cv2.TrackerMIL_create()
            elif algorithm == 'Median-Flow':
                tracker = cv2.TrackerMedianFlow_create()
            else:
                raise ValueError('Unknown algorithm type')
            bbox = select_bounding_box()
            tracker.init(FRAME, bbox)
        corner1 = int(bbox[0]), int(bbox[1])
        corner2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        distance = sqrt((((2.0 * bbox[0] + bbox[2]) / 2.0) - origin[0])**2
                        + (((2.0 * bbox[1] + bbox[3]) / 2.0) - origin[1])**2)
        time_points.append(frame_number / fps)
        dist_points.append(distance / scale)
        if not suppress_live_plot:
            plt.cla()
            plt.scatter(time_points, dist_points)
            plt.title(PLOT_TITLE)
            plt.xlabel(PLOT_XLABEL)
            plt.ylabel(PLOT_YLABEL)
            plt.pause(0.001)
        cv2.rectangle(FRAME, corner1, corner2, (255, 0, 0), 2, 1)
        cv2.imshow("Tracking", FRAME)
        key_press = cv2.waitKey(1) & 0xff
        if key_press == ESCAPE_KEY:
            break
    if not suppress_live_plot:
        plt.close()
        plt.ioff()
    cv2.destroyAllWindows()
    return np.array([time_points, dist_points]).T


def open_video(filepath):
    """Open the video file"""
    video = cv2.VideoCapture(filepath)
    if not video.isOpened():
        raise RuntimeError('Could not open specified video file')
    return video


def draw_line(event, x_press, y_press, flags, param): #pylint: disable=unused-argument
    """Draw a line on the frame"""
    global X_VAL_1, Y_VAL_1, X_VAL_2, Y_VAL_2, DRAWING, COPY
    if event == cv2.EVENT_LBUTTONDOWN:
        DRAWING = True
        X_VAL_1, Y_VAL_1 = x_press, y_press
    elif event == cv2.EVENT_MOUSEMOVE:
        if DRAWING and FRAME is not None:
            X_VAL_2, Y_VAL_2 = x_press, y_press
            COPY = FRAME.copy()
            cv2.line(COPY, (X_VAL_1, Y_VAL_1), (X_VAL_2, Y_VAL_2), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        DRAWING = False
        X_VAL_2, Y_VAL_2 = x_press, y_press
        cv2.line(COPY, (X_VAL_1, Y_VAL_1), (X_VAL_2, Y_VAL_2), (0, 255, 0), 2)


def parse_args():
    """Parse BigG args"""
    parser = argparse.ArgumentParser(
        description='Perform OpenCV object tracking on a video file.')
    parser.add_argument(
        '-a', '--algorithm',
        help='the tracking algorithm to use',
        choices=['KCF', 'MIL', 'Median-Flow'],
        default='KCF')
    parser.add_argument(
        'video_file',
        help='the video file containing the tracked object',
        metavar='VIDEO_FILE')
    parser.add_argument(
        '-o', '--output_file',
        help='output file into which to write NumPy data',
        default='bigG_out.npy')
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='overwrite the output file if it already exists')
    parser.add_argument(
        '--suppress_live_plot',
        action='store_true',
        help='suppress real-time plot of the tracked position')
    return parser.parse_args()


def press_enter():
    """Press ENTER to continue"""
    return input('Press ENTER to continue...')


# other globals
X_VAL_1, Y_VAL_1, X_VAL_2, Y_VAL_2 = 0, 0, 0, 0
DRAWING = False
FRAME = None
COPY = None
# messages
SET_SCALE_MSG = """
A window will pop up containing the first frame of the video. Please draw a
line on this window. This line should represent a known distance in the video
and will be used to set the scale. When you are happy with the scale line,
press ENTER to close the video frame."""
ROI_BOX_MSG = """
The first frame of the video will now be shown again. Please draw a box around
the object (ROI) you want to track."""
TRACKING_MSG = """
Object tracking will now begin. If the tracker loses the object, it will pop up
a new window for you to reselect the object. This may happen multiple times.
The entire process may take some time."""
TRACKING_MSG_W_PLOT = """
Object tracking will now begin. A plot will update live with the tracked
coordinates, taking into account both the scale and the frame rate of the video
file. If the tracker loses the object, it will pop up a new window for you to
reselect the object. This may happen multiple times. The entire process may
take some time."""
TRACKING_FAIL_MSG = """
The tracker lost the object. Please draw a box around the object you are
tracking"""
LAST_PLOT_MSG = """
Thanks for running the object tracking script. Your data points have been
saved. I will plot the data one last time. Close the plot when you are ready to
exit the program."""

if __name__ == "__main__":
    tracking()
