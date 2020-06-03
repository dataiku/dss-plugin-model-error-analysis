# the following class comes from https://github.com/arvkevi/kneed/blob/master/kneed/knee_locator.py
# modified for python 2.7
import numpy as np
from scipy import interpolate
from scipy.signal import argrelextrema
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from warnings import warn


class KneeLocator(object):
    def __init__(
        self,
        x,
        y,
        S = 1.0,
        direction = "increasing",
        interp_method = "interp1d",
        online = False,
    ):
        """
        Once instantiated, this class attempts to find the point of maximum
        curvature on a line. The knee is accessible via the `.knee` attribute.
        :param x: x values.
        :param y: y values.
        :param S: Sensitivity, original paper suggests default of 1.0
        :param direction: one of {"increasing", "decreasing"}
        :param interp_method: one of {"interp1d", "polynomial"}
        :param online: Will correct old knee points if True, will return first knee if False
        """
        # Step 0: Raw Input
        self.x = x
        self.y = y
        self.direction = direction
        self.S = S
        self.all_knees = set()
        self.all_norm_knees = set()
        self.all_knees_y = []
        self.all_norm_knees_y = []
        self.online = online

        # Step 1: fit a smooth line
        if interp_method == "interp1d":
            uspline = interpolate.interp1d(self.x, self.y)
            self.Ds_y = uspline(self.x)
        elif interp_method == "polynomial":
            pn_model = PolynomialFeatures(7)
            xpn = pn_model.fit_transform(self.x.reshape(-1, 1))
            regr_model = LinearRegression()
            regr_model.fit(xpn, self.y)
            self.Ds_y = regr_model.predict(
                pn_model.fit_transform(self.x.reshape(-1, 1))
            )
        else:
            warn(
                "{} is an invalid interp_method parameter, use either 'interp1d' or 'polynomial'".format(
                    interp_method
                )
            )
            return

        # Step 2: normalize values
        self.x_normalized = self.__normalize(self.x)
        self.y_normalized = self.__normalize(self.Ds_y)

        # Step 3: Calculate the Difference curve
        self.x_normalized, self.y_normalized = self.transform_xy(
            self.x_normalized, self.y_normalized, self.direction
        )
        # normalized difference curve
        self.y_difference = self.y_normalized - self.x_normalized
        self.x_difference = self.x_normalized.copy()

        # Step 4: Identify local maxima/minima
        # local maxima
        self.maxima_indices = argrelextrema(self.y_difference, np.greater_equal)[0]
        self.x_difference_maxima = self.x_difference[self.maxima_indices]
        self.y_difference_maxima = self.y_difference[self.maxima_indices]

        # local minima
        self.minima_indices = argrelextrema(self.y_difference, np.less_equal)[0]
        self.x_difference_minima = self.x_difference[self.minima_indices]
        self.y_difference_minima = self.y_difference[self.minima_indices]

        # Step 5: Calculate thresholds
        self.Tmx = self.y_difference_maxima - (
            self.S * np.abs(np.diff(self.x_normalized).mean())
        )

        # Step 6: find knee
        self.knee, self.norm_knee = self.find_knee()

        # Step 7: If we have a knee, extract data about it
        self.knee_y = self.norm_knee_y = None
        if self.knee:
            self.knee_y = self.y[self.x == self.knee][0]
            self.norm_knee_y = self.y_normalized[self.x_normalized == self.norm_knee][0]

    @staticmethod
    def __normalize(a):
        """normalize an array
        :param a: The array to normalize
        """
        return (a - min(a)) / (max(a) - min(a))

    @staticmethod
    def transform_xy(
        x, y, direction
    ):
        """transform x and y to concave, increasing based on given direction and curve"""
        # flip decreasing functions to increasing
        if direction == "decreasing":
            y = np.flip(y, axis=0)

        return x, y

    def find_knee(self):
        """This function finds and sets the knee value and the normalized knee value. """
        if not self.maxima_indices.size:
            warn(
                "No local maxima found in the difference curve\n"
                "The line is probably not polynomial, try plotting\n"
                "the difference curve with plt.plot(knee.x_difference, knee.y_difference)\n"
                "Also check that you aren't mistakenly setting the curve argument",
                RuntimeWarning,
            )
            return None, None

        # placeholder for which threshold region i is located in.
        maxima_threshold_index = 0
        minima_threshold_index = 0
        # traverse the difference curve
        for i, x in enumerate(self.x_difference):
            # skip points on the curve before the the first local maxima
            if i < self.maxima_indices[0]:
                continue

            j = i + 1

            # reached the end of the curve
            if x == 1.0:
                break

            # if we're at a local max, increment the maxima threshold index and continue
            if (self.maxima_indices == i).any():
                threshold = self.Tmx[maxima_threshold_index]
                threshold_index = i
                maxima_threshold_index += 1
            # values in difference curve are at or after a local minimum
            if (self.minima_indices == i).any():
                threshold = 0.0
                minima_threshold_index += 1

            if self.y_difference[j] < threshold:
                if self.direction == "decreasing":
                    knee = self.x[-(threshold_index + 1)]
                    norm_knee = self.x_normalized[-(threshold_index + 1)]
                else:
                    knee = self.x[threshold_index]
                    norm_knee = self.x_normalized[threshold_index]

                # add the y value at the knee
                y_at_knee = self.y[self.x == knee][0]
                y_norm_at_knee = self.y_normalized[self.x_normalized == norm_knee][0]
                if knee not in self.all_knees:
                    self.all_knees_y.append(y_at_knee)
                    self.all_norm_knees_y.append(y_norm_at_knee)

                # now add the knee
                self.all_knees.add(knee)
                self.all_norm_knees.add(norm_knee)

                # if detecting in offline mode, return the first knee found
                if self.online is False:
                    return knee, norm_knee

        if self.all_knees == set():
            warn("No knee/elbow found")
            return None, None

        return knee, norm_knee
