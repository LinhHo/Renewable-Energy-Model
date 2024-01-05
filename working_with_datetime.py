#! /usr/bin/env python
# -*- coding: latin-1 -*-
"""
Written in since 2014
by Christopher Frank and Andreas Anhaeuser
Insitute for Geophysics and Meteorology
University of Cologne
Germany
<andreasfrederik.anhaeuser@smail.uni-koeln.de>
"""

import numpy as np
import datetime as dt
import collections
import ephem
import pandas as pd
import calendar
from scipy.interpolate import griddata


#################################################
# CLASSES                                       #
#################################################
class DaytimePeriod:
    """A portion of the day cycle.
    
    This type is unaware of its absolute (calendar) date.
    It can extend beyond midnight, e. g. 23:00 to 01:00.
    
    Written in 2014
    by Andreas Anhaeuser
    Insitute for Geophysics and Meteorology
    University of Cologne
    Germany
    <andreasfrederik.anhaeuser@smail.uni-koeln.de> """

    def __init__(self, dt_start=None, dt_end=None, allow_whole_day=True):
        """dt_start and dt_end must be dt.datetime or dt.time objects.
        
        The absolute calendar dates of the input arguments are ignored. This
        means that calling this function with dt_start 1st Jan 1970 23:00 and
        dt_end 30th Mar 1970 01:00 is equivalent to calling it with dt_start
        1st Jan 2014 23:00 and dt_end 30th Mar 1789 01:00.  Both will result in
        a time period between 23:00 and 01:00.
        
        Boundaries: The lower end dt_start is inclusive, the upper end dt_end
        is exclusive.

        Parameters
        ----------
        dt_start : datetime.datetime or datetime.time object
                   default: midnight
        dt_end : datetime.datetime or datetime.time object
                 default: midnight
        allow_whole_day : boolean
                 applies only if start and end are equal (or equivalent)
                 see Note below
        
        Note
        ----
        In case, dt_start == dt_end
        * if allow_whole_day==True, the time period will contain the whole day.
        * if allow_whole_day==False, the time period will not contain anything.
        """
        #####################################
        # DEFAULT                           #
        #####################################
        if dt_start is None:
            dt_start = dt.datetime(1, 1, 1)
        if dt_end is None:
            dt_end = dt.datetime(1, 1, 1)

        #####################################
        # INPUT CHECK                       #
        #####################################
        for d in [dt_start, dt_end]:
            if d.__class__ not in [dt.datetime, dt.time]:
                raise TypeError('dt_start and dt_end must be instances of ' +
                        'datetime. datetime or datetime.time.')  
        if allow_whole_day.__class__ is not bool:
            raise TypeError('allow_whole_day must be a boolean.')
        
        ####################################
        # CONVERT TO dt.datetime           #
        ####################################
        if dt_start.__class__ is dt.time:
            start = dt.datetime.combine(dt.date(1, 1, 1), dt_start)
        else:
            start = dt_start
        if dt_end.__class__ is dt.time:
            end = dt.datetime.combine(dt.date(1, 1, 1), dt_end)
        else:
            end = dt_end   
            
        ####################################
        # SHIFT TO YEAR 1                  #
        ####################################
        """
        self.start will be on Jan 1st 0001 CE.
        self.end will be between 0 and 1 day later than self.end, all will thus
        be on Jan 1st or Jan 2nd in year 0001 CE.
        """
        start = start.replace(1, 1, 1)
        end   = end.replace(1, 1, 1)

        ####################################
        # CHECK SEQUENCE                   #
        ####################################
        # make sure that end is not earlier that start:
        if end < start:
            end = end.replace(day=2)
        if end == start and allow_whole_day:
            end = end.replace(day=2)

        ####################################
        # CREATE FIELDS                    #
        ####################################
        self.start = start
        self.end   = end
        
    def length(self):
        """Return an instance of datetime.timedelta."""
        return self.end - self.start        
        
    def contains(self, d):
        """Check whether d is within Season or not.
        
        Returns a boolean."""
        ####################################
        # INPUT CHECK                      #
        ####################################
        if d.__class__ not in [dt.datetime, dt.time]:
            raise TypeError('Argument must be an instance of ' +
                    'datetime. datetime or datetime.time.')  

        ####################################
        # CONVERT TO dt.datetime           #
        ####################################
        if d.__class__ is dt.time:
            dd = dt.datetime.combine(dt.date(1, 1, 1),d)
        else:
            dd = d.replace(1, 1, 1)
            
        ####################################
        # CHECK RELATIVE POSITIONS         #
        ####################################
        # make sure that dd is later than self.start:
        if dd < self.start:
            dd = dd.replace(day=2)
        # check whether dd is earlier than self.end:
        if dd < self.end:
            return True
        else:
            return False        
        
class Season:
    """A section of the year cycle.
        
       This type is unaware of its absolute year number.  It can extend beyond
       New Year, e. g. Nov 2nd to Jan 10th.    
       
       Written in 2014 by Andreas Anhaeuser Insitute for Geophysics and
       Meteorology University of Cologne Germany
       <andreasfrederik.anhaeuser@smail.uni-koeln.de> """
    def __init__(
        self,
        dt_start=dt.datetime(1, 1, 1),
        dt_end=dt.datetime(1, 1, 1),
        months='',
        allow_whole_year=True
        ):
        """dt_start and dt_end must be dt.datetime or dt.date objects.
        
        The absolute year numbers of the input arguments are ignored. This
        means that calling this function with dt_start 1st Jan 1970 and dt_end
        30th Mar 1970 is equivalent to calling it with dt_start 1st Jan 2014
        and dt_end 30th Mar 1789.
        
        BOUNDARIES:
        The lower end dt_start is inclusive, the upper end dt_end is exclusive.
        
        SPECIAL CASE: dt_start == dt_end
        * if allow_whole_year==True, the season will contain the whole year.
        * if allow_whole_year==False, the season will not contain anything.
        
        SPECIAL CASE: February 29th
        If the date of dt_start and/or dt_end is Feb 29th, they will be
        treated as Mar 1st 00:00.
        """
        #### CHECK INPUT ###
        for d in [dt_start, dt_end]:
            if d.__class__ not in [dt.datetime, dt.date]:
                raise TypeError('dt_start and dt_end must be instances of ' +
                        'datetime. datetime or datetime.date.')  
        if allow_whole_year.__class__ is not bool:
            raise TypeError('allow_whole_year must be a boolean.')
        # months:
        if months.lower() == 'year':
            mon = 'jfmamjjasond'
        else:
            mon = months.lower()
        allowed_month_seqs = 'jfmamjjasondjfmamjjasond'
        allowed_months =[
                'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                ]
        if mon.__class__ is not str:
            raise TypeError('months must be a string.')
        if len(mon) == 1:
            raise ValueError('months must be a string of at least two ' +
                'month initial letters.')
        if len(mon) > 12:
            raise ValueError('months must not contain more than 12 letters.')
        if (mon not in allowed_months and 
                mon[:3].lower() not in allowed_month_seqs):
            raise ValueError('months is not an allowed sequence of ' +
                'month initial letters.')
        # self.start will be in year 1
        # self.end will be between 0 and 1 year later than self.end, and will
        # thus be in year 1 or 2
        if mon == '':
            start = year_one(dt_start)
            end   = year_one(dt_end)
        elif mon in allowed_month_seqs:
            first_month = allowed_month_seqs.find(mon) + 1
            last_month  = (first_month + len(mon)) % 12
            if last_month == 0:
                last_month = 12
            start = dt.datetime(1, first_month, 1)
            end   = dt.datetime(1, last_month, 1)
        elif mon[:3].lower() in allowed_months:
            first_month = allowed_months.index(mon) + 1
            last_month  = (first_month + 1) % 12
            if last_month == 0:
                last_month = 12
            start = dt.datetime(1, first_month, 1)
            end   = dt.datetime(1, last_month, 1)
            
            
        # make sure that end is not earlier that start:
        if end < start:
            end = end.replace(year=2)
        if end == start and allow_whole_year:
            end = end.replace(year=2)
            
        self.start = start
        self.end   = end
    def length(self):
        """Return an instance of datetime.timedelta."""
        return self.end - self.start
    def contains(self, d):
        """Check whether d is within Season or not.
        
        Returns a boolean."""
        #### CHECK INPUT ###
        if d.__class__ not in [dt.datetime, dt.date]:
            raise TypeError('Argument must be an instance of ' +
                    'datetime. datetime or datetime.date.')  

        dd = year_one(d)
        if dd < self.start:
            dd = dd.replace(year=2)
            
        # check whether dd is earlier than self.end:
        if dd < self.end:
            return True
        else:
            return False

class DateTime(dt.datetime):
    @classmethod
    def from_datetime(cls, d):
        D = d.date()
        T = d.time()
        return cls.combine(D, T)

    @classmethod
    def from_seconds(cls, s, ref=dt.datetime(1970, 1, 1)):
        """Convert integer into datetime.
       
        Parameters
        ----------
        s : a number
            seconds since ref
        ref : datetime.datetime object
              reference date
        
        Returns
        -------
        a datetime.datetime object
        """
        if not isinstance(ref, dt.datetime):
            raise TypeError('ref must be a datetime.datetime object.')
        d = ref + dt.timedelta(seconds=float(s))
        return cls.from_datetime(d)

    def to_seconds(self, ref=dt.datetime(1970, 1, 1)):
        """Convert datetime into integer.
       
        Parameters
        ----------
        ref : dt.datetime object
              reference date

        Returns
        -------
        float : seconds since ref
        """
        if ref.__class__ is not dt.datetime:
            raise TypeError('ref must be a datetime.datetime object.')
        diff = self - ref  # a dt.timedelta object
        return diff.total_seconds()


#################################################
# FUNCTIONS                                     #
#################################################
def year_one(d):
    """Return a datetime.datetime object in year 1 CE.
    
    Parameters
    ----------
    d : an instance of datetime.datetime or datetime.date

    Returns
    -------
    A datetime.datetime object. Date and time is be the same as of the
    input object, only the year is set to 1.
    
    Note
    ----
    Dates on 29th Feb will are converted to 1st Mar 00:00.
    """
    # special case: Feb 29th:
    if d.month == 2 and d.day == 29:
        return dt.datetime(1, 3, 1)
    # datetime.date object:
    if d.__class__ is dt.date:
        return dt.datetime.combine(d.replace(year=1), dt.time())
    # datetime.datetime object:
    if d.__class__ is dt.datetime:
        return d.replace(year=1)

def utc_sunrise_sunset(d, lon, lat, alt=0):
    """Return SR and SS for this day as instances of datetime.datetime.
        
    Parameters
    ----------
    d : datetime.date or datetime.datetime object.
    lon: a numerical denoting the observer's longitude in deg North
    lat: a numerical denoting the observer's latitude in deg East
    alt: a numerical denoting the observer's altitude above sea level in meters
    
    Returns
    -------
    (SR, SS) where SR and SS are datetime.datetime objects or None.

    SR and SS denote sun rise and sun set of the day specified by d.
    If (in polar regions,) there is no sun set or sun rise at this day, SR
    and/or SS are None.
    
    Note
    ----
    All times and dates are treated as UTC (not local time). This means that SS
    can be before SR (this is roughly the case for places that lie on the
    hemisphere where abs(lon) > 90).
    
    Written in 2014
    by Andreas Anhaeuser
    Insitute for Geophysics and Meteorology
    University of Cologne
    Germany
    <andreasfrederik.anhaeuser@smail.uni-koeln.de> """
    #### CHECK INPUT ###
    if d.__class__ not in [dt.date, dt.datetime]:
        raise TypeError(
        'd must be an instance of datetime.date or datetime.datetime.')
    for num in [lon, lat, alt]:
        if num.__class__ not in [
            int, float, np.int, np.float16, np.float32, np.float64
            ]:
            raise TypeError('lat, lon and alt must be numerical values.')
    if not -180 <= lon <= 180:
        raise TypeError('lon must be between -180 and 180.')
    if not -90 <= lat <= 90:
        raise TypeError('lat must be between -90 and 90.')
    #### CONVERSIONS ###
    # convert date to an instance of datetime.date:
    if d.__class__ is dt.datetime:
        dd = d.date()
    else:
        dd = d
    #### CREATE EPHEM OBJECTS ###
    # create Sun object:
    sun = ephem.Sun(dd)
    # create observer object:
    site = ephem.Observer()
    site.lon = str(lon)
    site.lat = str(lat)
    site.elevation = alt
    site.date = dd
    #### GET SR AND SS ###
    try:
        SR = site.next_rising(sun).datetime()
    except ephem.NeverUpError:
        SR = None
    except ephem.AlwaysUpError:
        SR = None
    try:
        SS = site.next_setting(sun).datetime()
    except ephem.NeverUpError:
        SS = None
    except ephem.AlwaysUpError:
        SS = None
   
    #### RETURN ###
    return (SR, SS)
   
def datetime_range(
        start,
        end,
        increment,
        season_of_year = None,
        daytime_period = None,
        ):
    """Return a list of instances of datetime.
        
    Parameters
    ----------
    * start, end and increment must be instances of datetime.datetime.
    * season_of_year must be None or an instance of Season.
    * daytime_period must be None or an instance of DaytimePeriod.
    If season_of_year and/or daytime_period are None, they are interpreted
    as whole year and whole day cycles, respectively.
    
    The function works similar to the standard range() function and is
    intended to be an extension of it to datetime objects.
    
    The returned list contains only elements that that lie within
    season_of_year and daytime_period.
    
    start is inclusive, end is exclusive.
    
    Written in 2014
    by Andreas Anhaeuser
    Insitute for Geophysics and Meteorology
    University of Cologne
    Germany
    <andreasfrederik.anhaeuser@smail.uni-koeln.de> """
    #### INPUT CHECK: ###
    for x in [start, end]:
        if x.__class__ is not dt.datetime:
            raise TypeError(
                'start and end must be instances of datetime.datetime.')
    if increment.__class__ is not dt.timedelta:
        raise TypeError(
            'increment must be an instance of datetime.timedelta.')
    if season_of_year is not None and season_of_year.__class__ is not Season:
        raise TypeError(
            'season_of_year must be None or an instance of Season.')
    if (daytime_period is not None and 
            daytime_period.__class__ is not DaytimePeriod):
        raise TypeError(
            'daytime_period must be None or an instance of DaytimePeriod.')
    #
    # convert season_of_year and daytime_period if neccessary:
    if season_of_year is None:
        # whole year:
        season = Season(dt.datetime(1,1,1), dt.datetime(1,1,1))
    else:
        season = season_of_year
    if daytime_period is None:
        # whole day:
        timeperiod = DaytimePeriod(dt.time(), dt.time())
    else:
        timeperiod = daytime_period
        
    out = []
    d = start
    while d < end:
        if season.contains(d) and timeperiod.contains(d):
            out.append(d)
        d += increment
    return out

def datetime_to_seconds(
        d,
        reference_date=dt.datetime(1970,1,1),
        ):
    """Convert datetime into integer.
   
    INPUT:
    d: datetime_list is a list of instances of datetime.
    reference_date: is an instance of datetime (default 01.01.1970).
    
    
    OUTPUT:
    The function return a list of numerical values which are the seconds since
    reference_date.
    
    Written in 2014
    by Andreas Anhaeuser
    Insitute for Geophysics and Meteorology
    University of Cologne
    Germany
    <andreasfrederik.anhaeuser@smail.uni-koeln.de> """
    if isinstance(d, collections.Iterable):
        if not any (d):
            return []
        else:
            return [datetime_to_seconds(dd, reference_date=reference_date)
                    for dd in d]
    distance = d - reference_date
    # (this is an instance of timedelta)
    return distance.total_seconds()
    
def seconds_to_datetime(
        seconds,
        reference_date=dt.datetime(1970,1,1),
        ):
    """Convert integer into datetime.
   
    Parameters
    ----------
    datetime_list is a list of instances of datetime.
    reference_date is an instance of datetime.
    
    Returns
    -------
    The function return a list of numerical values which are the seconds since
    reference_date.
    
    Written in 2014
    by Andreas Anhaeuser
    Insitute for Geophysics and Meteorology
    University of Cologne
    Germany
    <andreasfrederik.anhaeuser@smail.uni-koeln.de> """
    if isinstance(seconds, collections.Iterable):
        return([seconds_to_datetime(s, reference_date) for s in seconds])
    return reference_date + dt.timedelta(seconds=float(seconds))

def julian_days_to_datetime(days):
    """Return a datetime.datetime object."""
    if isinstance(days, collections.Iterable):
        return([julian_days_to_datetime(d) for d in days])
    JD0 = 1721425.5  # Julian date of 1st Jan 1, 00:00
    return dt.datetime(1, 1, 1) + dt.timedelta(days=days - JD0)
   
def datetime_to_julian_days(time):
    """Return a float."""
    if isinstance(time, collections.Iterable):
        return([datetime_to_julian_days(t) for t in time])
    JD0 = 1721425.5  # Julian date of 1st Jan 1, 00:00
    diff = (time - dt.datetime(1, 1, 1))
    D  = diff.days
    S  = diff.seconds / 86400.
    MS = diff.microseconds / (86400 * 1e6)
    return D + S + MS + JD0

def hourdotfractional_to_seconds(vec):
    '''
    Calculates the seconds since beginning of day.
    
    INPUT
    -----
    vec: np.array
         Array of hour.fractional 
    OUTPUT
    sec_sum: np.array
         Array of seconds since the beginning of the day
    '''
    hour = vec.astype(int)
    min_frac = array([h-(h.astype(int)) for h in vec]) * 60.
    min = min_frac.astype(int)
    sec = array([h-(h.astype(int)) for h in min_frac]) * 60.
    #sec = sec_frac.astype(int)
    sec_sum = sec + min*60 + hour*60*60
    return sec_sum


def daytimediff(minuend, subtrahend, mode=None):
    """Return an instance of datetime.timedelta.
    
    minuend and subtrahend must be instances of datetime.datime or
    datetime.time
    
    mode: (None, 'abs', 'pos', 'neg')
    * None: a value between -12h and + 12h is returned
    * 'abs': a value between 0 and 12h is returned. The absolute difference.
    * 'pos': a value between 0 and 24h is returned.
    * 'neg': a value between -24h and 0 is returned.    
    """
    
    if minuend.__class__ is list:
        return [daytimediff(m, subtrahend) for m in minuend]
    if subtrahend.__class__ is list:
        return [daytimediff(minuend, s) for s in subtrahend]
    
    for d in [minuend, subtrahend]:
        if d.__class__ not in [dt.datetime, dt.time]:
            raise TypeError(
          'Arguments must be instances of datetime.datime or datetime.time.')
          
    if minuend.__class__ is dt.datetime:
        m = minuend.replace(1, 1, 1)
    else:
        hour = minuend.hour
        minute = minuend.minute
        second = minuend.second
        microsec = minuend.microsecond
        m = dt.datetime(1, 1, 1, hour, minute, second, microsecond)
        
    if subtrahend.__class__ is dt.datetime:
        s = subtrahend.replace(1, 1, 1)
    else:
        hour = subtrahend.hour
        minute = subtrahend.minute
        second = subtrahend.second
        microsec = subtrahend.microsecond
        s = dt.datetime(1, 1, 1, hour, minute, second, microsecond)      
    
    diff = m - s
    if mode is None:
        while diff.total_seconds() > 86400/2:
            diff -= dt.timedelta(days=1)
        while diff.total_seconds() < -86400/2:
            diff += dt.timedelta(days=1)   
    
    if mode == 'abs':
        diff = abs(diff, -diff)

    if mode == 'pos':
        while diff.total_seconds() < 0:
            diff += dt.timedelta(days=1)

    if mode == 'neg':
        while diff.total_seconds() > 0:
            diff -= dt.timedelta(days=1)
        
    return diff
        
def next_month(year, month):
    if month < 12:
        month += 1
    else:
        month = 1
        year += 1
    return year, month

def next_day(year, month, day):
    thisday = dt.datetime(year, month, day)
    nextday = thisday + dt.timedelta(days=1)
    year  = nextday.year
    month = nextday.month
    day   = nextday.day
    return year, month, day


###################################################
# DAY OF YEAR                                     #
###################################################
def doy(date):
    """Return day of year and year as pair of ints.

        Parameters
        ----------
        date : datetime.date or datetime.datetime

        Returns
        -------
        doy : int
            day of year. Between 1 and 366 (inclusive, in leap years)
        year : int

        Note
        ----
        1st January is DOY 1 (not 0).
    """
    if isinstance(date, dt.datetime):
        date = date.date()
    year = date.year
    ref = dt.date(year, 1, 1)
    doy = (date - ref).days + 1
    return doy, year

def date_from_doy(doy, year):
    """Return a datetime.date.

        Parameters
        ----------
        doy : int
            day of year. Can be negative or positve. Can be larger than 366.
        year : int

        Returns
        -------
        date : datetime.date or datetime.datetime

        Note
        ----
        1st January is DOY 1 (not 0).
    """
    return dt.date(year, 1, 1) + dt.timedelta(days=doy-1)

def datetime_from_doy(doy, year):
    """Return a datetime.date.

        Parameters
        ----------
        doy : int
            day of year. Can be negative or positve. Can be larger than 366.
        year : int

        Returns
        -------
        date : datetime.date or datetime.datetime

        Note
        ----
        1st January is DOY 1 (not 0).
    """
    return dt.datetime(year, 1, 1) + dt.timedelta(days=doy-1)


###################################################
# sun / day / night                               #
###################################################
def utc_sunrise_sunset(d, lon, lat, alt=0, pres=None, temp=None):
    """Return SR and SS for this day as a pair of dt.datetime.

        The time of day of the input is ignored, only the date is taken into
        account.

        The horizon is assumed to be at 0 degrees, regardless of the altitude.

        Parameters
        ----------
        d : datetime.date or datetime.datetime
        lon: float
            the observer's longitude in deg North
        lat: float
            the observer's latitude in deg East
        alt: float
            the observer's altitude above the surface in meters
        pres : float, optional
            (Pa) pressure at ground. Used to compute atmospheric light
            refraction. Note: overrides alt !
        temp : float, optional
            (K) temperature at ground. Used to compute atmospheric light
            refraction. Default: 288.15.

        Returns
        -------
        SR, SS : datetime.datetime or None
            SR and SS denote sun rise and sun set of the day specified by d.
            If (in polar regions,) there is no sun set or sun rise at this day,
            SR and/or SS are None.

        Notes
        -----
        All times and dates are treated as UTC (not local time). This means
        that SS can be before SR (this is roughly the case for places that lie
        on the hemisphere where abs(lon) > 90).

        SR and SS as returned by this function are always on the same UTC date
        as d. If such a SR and/or SS does not exist, then the respective value
        is set to None. This may be counter-intuitive in some cases, e. g. on a
        day where the previous sun rise is just before 0:00 and the following
        is just after 0:00 (i. e. 24 h and some minutes later), then on the
        actual day, there is no sun rise and None is returned:

              | day (n-1) | day n   | day (n+1)
        ------+-----------+---------+-----------
        SR    | 23:59     | ----    | 00:01
        SS    | 11:20     | 11:18   | 11:16

        Dependencies
        ------------
        Makes use of the package ephem.

        Raises
        ------
        TypeError, ValueError

        Tested
        ------
        Moderately tested. Seems bug-free, but not 100% sure.

        Author
        ------
        Written in 2014-2016
        by Andreas Anhaeuser
        Insitute for Geophysics and Meteorology
        University of Cologne
        Germany
        <anhaeus@meteo.uni-koeln.de>
    """
    #############################################
    # INPUT CHECK                               #
    #############################################
    if not isinstance(d, dt.date):
        raise TypeError('d must be an instance of datetime.date.')
    if not -180 <= lon <= 180:
        raise ValueError('lon must be between -180 and 180.')
    if not -90 <= lat <= 90:
        raise ValueError('lat must be between -90 and 90.')

    #############################################
    # SET TIME TO MIDNIGHT                      #
    #############################################
    if isinstance(d, dt.datetime):
        d = d.date()

    #############################################
    # CREATE EPHEM OBJECTS                      #
    #############################################
    # create Sun object:
    sun = ephem.Sun(d)

    # create observer object:
    site = ephem.Observer()
    site.lon = str(lon)
    site.lat = str(lat)
    site.elevation = alt
    if pres is None:
        site.compute_pressure()
    else:
        site.pressure = pres * 1e-2  # (convert from Pa to hPa)
    if temp is not None:
        site.temp = temp - 273.15    # (convert from deg C to K)
    site.date = d

    #############################################
    # SR AND SS                                 #
    #############################################
    try:
        SR = site.next_rising(sun).datetime()
        # make sure SR is on the same day:
        if SR.date() != d:
            SR = None
    except ephem.NeverUpError:
        SR = None
    except ephem.AlwaysUpError:
        SR = None

    try:
        SS = site.next_setting(sun).datetime()
        # make sure SS is on the same day:
        if SS.date() != d:
            SS = None
    except ephem.NeverUpError:
        SS = None
    except ephem.AlwaysUpError:
        SS = None

    return (SR, SS)

def is_day(d, lon, lat, alt=0., pres=None, temp=None):
    """Return a bool.

        Consistent with utc_sunrise_sunset within tens of microseconds.

        Parameters
        ----------
        d : datetime.date or datetime.datetime
            UTC
        lon: float
            the observer's longitude in deg North
        lat: float
            the observer's latitude in deg East
        alt: float
            the observer's altitude above the surface in meters
        pres : float, optional
            (Pa) pressure at ground. Used to compute atmospheric light
            refraction. Note: overrides alt !
        temp : float, optional
            (K) temperature at ground. Used to compute atmospheric light
            refraction. Default: 288.15.

        Returns
        -------
        bool

        Dependencies
        ------------
        Makes use of the package ephem.

        Raises
        ------
        TypeError, ValueError

        Tested
        ------
        Moderately tested. Seems bug-free, but not 100% sure.

        Author
        ------
        Written in 2016
        by Andreas Anhaeuser
        Insitute for Geophysics and Meteorology
        University of Cologne
        Germany
        <anhaeus@meteo.uni-koeln.de>
    """
    #############################################
    # INPUT CHECK                               #
    #############################################
    if not isinstance(d, dt.datetime):
        raise TypeError('d must be datetime.datetime.')
    if not -180 <= lon <= 180:
        raise ValueError('lon must be between -180 and 180.')
    if not -90 <= lat <= 90:
        raise ValueError('lat must be between -90 and 90.')

    #############################################
    # CREATE EPHEM OBJECTS                      #
    #############################################
    # create Sun object:
    sun = ephem.Sun(d)

    # create observer object:
    site = ephem.Observer()
    site.lon = str(lon)
    site.lat = str(lat)
    site.elevation = alt
    site.date = d
    if pres is None:
        site.compute_pressure()
    else:
        site.pressure = pres * 1e-2  # (convert from Pa to hPa)
    if temp is not None:
        site.temp = temp - 273.15    # (convert from deg C to K)

    # compute sun elevation
    sun.compute(site)
    elevation = sun.alt

    # take into account extent of the sun:
    size_arcsec = sun.size
    size = size_arcsec *np.pi / (3600 *180)
    elev_top = elevation + size / 2
    return elev_top >= 0

def is_night(d, lon, lat):
    """Return a bool.

    Description, see `is_day`."""
    return not is_day(d, lon, lat)

def last_sunrise(d, lon, lat, alt=0, pres=None, temp=None):
    """Return a dt.datetime.

        The horizon is assumed to be at 0 degrees, regardless of the altitude.

        Parameters
        ----------
        d : datetime.date or datetime.datetime
        lon: float
            the observer's longitude in deg North
        lat: float
            the observer's latitude in deg East
        alt: float
            the observer's altitude above the surface in meters
        pres : float, optional
            (Pa) pressure at ground. Used to compute atmospheric light
            refraction. Note: overrides alt !
        temp : float, optional
            (K) temperature at ground. Used to compute atmospheric light
            refraction. Default: 288.15.

        Returns
        -------
        datetime.datetime

        Notes
        -----
        Unlike utc_sunrise_sunset, this function always returns a
        datetime.datetime object. If necessary, it goes back several days until
        it finds a sunrise.

        Dependencies
        ------------
        Makes use of the package ephem.

        Tested
        ------
        Moderately tested. Seems bug-free, but not 100% sure.

        Author
        ------
        Written in 2016
        by Andreas Anhaeuser
        Insitute for Geophysics and Meteorology
        University of Cologne
        Germany
        <anhaeus@meteo.uni-koeln.de>
    """
    # Idea
    # ====
    # 1. If sunrise on this day is earlier than d, return it.
    # 2. If sunrise is later than d, go back one day.
    # 3. In polar regions, it may be necessary to go back several days to find
    #    a sun rise, hence the while-loop.

    ###################################################
    # CREATE SUN OBJECT                               #
    ###################################################
    sun = ephem.Sun(d)

    ###################################################
    # CREATE OBSERVER OBJECT                          #
    ###################################################
    site = ephem.Observer()
    site.lon = str(lon)
    site.lat = str(lat)
    site.elevation = alt
    if pres is None:
        site.compute_pressure()
    else:
        site.pressure = pres * 1e-2  # Pa --> hPa
    if temp is not None:
        site.temp = temp - 273.15    # K --> deg C
    site.date = d

    ###################################################
    # FIND SUN RISE                                   #
    ###################################################
    # in extreme cases (close to the pole), the last sun rise may be up to half
    # a year ago (<= 183 days).
    d_try = d
    d_inc = dt.timedelta(days=1)
    count = 0
    found = False
    while not found:
        try:
            SR = site.previous_rising(sun).datetime()
            found = True
        except ephem.NeverUpError:
            pass
        except ephem.AlwaysUpError:
            pass
        count += 1
        d_try = d_try - d_inc
        assert count < 184        # (d) half a year
    return SR

def last_sunset(d, lon, lat, alt=0, pres=None, temp=None):
    """Return a dt.datetime.

    For description, see last_sunrise
    """
    # for comments, see last_sunrise.

    ###################################################
    # CREATE SUN OBJECT                               #
    ###################################################
    sun = ephem.Sun(d)

    ###################################################
    # CREATE OBSERVER OBJECT                          #
    ###################################################
    site = ephem.Observer()
    site.lon = str(lon)
    site.lat = str(lat)
    site.elevation = alt
    if pres is None:
        site.compute_pressure()
    else:
        site.pressure = pres * 1e-2
    if temp is not None:
        site.temp = temp - 273.15
    site.date = d

    ###################################################
    # FIND SUN RISE                                   #
    ###################################################
    d_try = d
    d_inc = dt.timedelta(days=1)
    count = 0
    found = False
    while not found:
        try:
            event = site.previous_setting(sun).datetime()
            found = True
        except ephem.NeverUpError:
            pass
        except ephem.AlwaysUpError:
            pass
        count += 1
        d_try = d_try - d_inc
        assert count < 184
    return event

def next_sunrise(d, lon, lat, alt=0, pres=None, temp=None):
    """Return a dt.datetime.

    For description, see last_sunrise
    """
    # for comments, see last_sunrise.

    ###################################################
    # CREATE SUN OBJECT                               #
    ###################################################
    sun = ephem.Sun(d)

    ###################################################
    # CREATE OBSERVER OBJECT                          #
    ###################################################
    site = ephem.Observer()
    site.lon = str(lon)
    site.lat = str(lat)
    site.elevation = alt
    if pres is None:
        site.compute_pressure()
    else:
        site.pressure = pres * 1e-2
    if temp is not None:
        site.temp = temp - 273.15
    site.date = d

    ###################################################
    # FIND SUN RISE                                   #
    ###################################################
    d_try = d
    d_inc = dt.timedelta(days=1)
    count = 0
    found = False
    while not found:
        try:
            event = site.next_rising(sun).datetime()
            found = True
        except ephem.NeverUpError:
            pass
        except ephem.AlwaysUpError:
            pass
        count += 1
        d_try = d_try - d_inc
        assert count < 184
    return event

def next_sunset(d, lon, lat, alt=0, pres=None, temp=None):
    """Return a dt.datetime.

    For description, see last_sunrise
    """
    # For comments, see last_sunrise

    ###################################################
    # CREATE SUN OBJECT                               #
    ###################################################
    sun = ephem.Sun(d)

    ###################################################
    # CREATE OBSERVER OBJECT                          #
    ###################################################
    site = ephem.Observer()
    site.lon = str(lon)
    site.lat = str(lat)
    site.elevation = alt
    if pres is None:
        site.compute_pressure()
    else:
        site.pressure = pres * 1e-2
    if temp is not None:
        site.temp = temp - 273.15
    site.date = d

    ###################################################
    # FIND SUN RISE                                   #
    ###################################################
    d_try = d
    d_inc = dt.timedelta(days=1)
    count = 0
    found = False
    while not found:
        try:
            event = site.next_setting(sun).datetime()
            found = True
        except ephem.NeverUpError:
            pass
        except ephem.AlwaysUpError:
            pass
        count += 1
        d_try = d_try - d_inc
        assert count < 184
    return event


def sun_position(d,lon,lat):
    '''
    Claculate elevation(=altitude) and azimuth
    
    For those unfamiliar with azimuth and altitude: T
    hey describe position in the sky by measuring angle 
    around the horizon, then angle above the horizon.
    
    Parameters
    ----------
    d : datetime.date or datetime.datetime
    lon: float
        the observer's longitude in deg North
    lat: float
        the observer's latitude in deg East
    
    Returns
    -------
    altitude: float
    azimuth: float

    
    '''
    ###################################################
    # CREATE SUN OBJECT                               #
    ###################################################
    sun = ephem.Sun(d)

    ###################################################
    # CREATE OBSERVER OBJECT                          #
    ###################################################
    site = ephem.Observer()
    site.lon = str(lon)
    site.lat = str(lat)
    site.date = d

    sun.compute(site)
    altitude_rad = sun.alt #radiant
    azimuth_rad = sun.az   #radiant
    altitude_grad = 360. / (2*np.pi) * altitude_rad
    azimuth_grad = 360. / (2*np.pi) * azimuth_rad
    return altitude_grad, azimuth_grad


def get_sun_position_of_matrix_without_temporal_interpo(DJF_dates_all, lon_obs_sub, lat_obs_sub, azimuth_angle_calc=True):
    '''
    This function calculates the elevation- and azimuth angle of the sun position.
    For computational costs the angles is calculated every 4th time stamp and every 3rd grid.
    The rest is linearly interpolated.
    
    INPUT 
    -----
    DJF_dates_all: dt.datetime
                   Dates of interest (first dimension of matrix)
    lon_obs_sub: float matrix
    lat_obs_sub: float matrix
    
    OUTPUT
    ------
    elevation_angle: float matrix
    azimuth_angle: float matrix
    '''
    angles = np.nan * np.empty( (len(DJF_dates_all), 
                                          lon_obs_sub.shape[0], 
                                          lon_obs_sub.shape[1],
                                          2), dtype = np.float32 )
    elevation_angle = np.nan * np.empty( (len(DJF_dates_all), 
                                          lon_obs_sub.shape[0], 
                                          lon_obs_sub.shape[1]) ,dtype=np.float32) 
    if azimuth_angle_calc:
        azimuth_angle = np.nan * np.empty( (len(DJF_dates_all), 
                                              lon_obs_sub.shape[0], 
                                              lon_obs_sub.shape[1]) , dtype=np.float32)
    for (i,j) in enumerate(np.arange(0,len(DJF_dates_all),1)):
        print j
        for i_lat in np.append(np.arange(0,lon_obs_sub.shape[0],3),[lon_obs_sub.shape[0]-1]):
            for i_lon in np.append(np.arange(0,lon_obs_sub.shape[1],3),[lon_obs_sub.shape[1]-1]):
                angles[j,i_lat,i_lon] = sun_position(DJF_dates_all[j],
                                          lon_obs_sub[i_lat,i_lon],
                                          lat_obs_sub[i_lat,i_lon])

        elevation_angle[j] = angles[j,:,:,0]
        if azimuth_angle_calc:
            azimuth_angle[j] = angles[j,:,:,1]
            azimuth_angle[j] = fill_nans(azimuth_angle[j])
        elevation_angle[j] = fill_nans(elevation_angle[j])
    
    #for i in np.arange(lon_obs_sub.shape[1]):
    #    # Spatial interpolation
    #    print i
    #    elevation_angle[:,:,i] = fill_nans(elevation_angle[:,:,i])
    #    if azimuth_angle_calc:
    #        azimuth_angle[:,:,i] = fill_nans(azimuth_angle[:,:,i])
    
    if azimuth_angle_calc:
        return elevation_angle, azimuth_angle
    else:
        return elevation_angle



def get_sun_position_of_matrix(DJF_dates_all, lon_obs_sub, lat_obs_sub, azimuth_angle_calc=True):
    '''
    This function calculates the elevation- and azimuth angle of the sun position.
    For computational costs the angles is calculated every 4th time stamp and every 3rd grid.
    The rest is linearly interpolated.
    
    INPUT 
    -----
    DJF_dates_all: dt.datetime
                   Dates of interest (first dimension of matrix)
    lon_obs_sub: float matrix
    lat_obs_sub: float matrix
    
    OUTPUT
    ------
    elevation_angle: float matrix
    azimuth_angle: float matrix
    '''
    angles = np.nan * np.empty( (len(DJF_dates_all), 
                                          lon_obs_sub.shape[0], 
                                          lon_obs_sub.shape[1],
                                          2) )
    elevation_angle = np.nan * np.empty( (len(DJF_dates_all), 
                                          lon_obs_sub.shape[0], 
                                          lon_obs_sub.shape[1]) ) 
    if azimuth_angle_calc:
        azimuth_angle = np.nan * np.empty( (len(DJF_dates_all), 
                                              lon_obs_sub.shape[0], 
                                              lon_obs_sub.shape[1]) )
    for (i,j) in enumerate(np.append(np.arange(0,len(DJF_dates_all),4),len(DJF_dates_all)-1)):
        print j
        for i_lat in np.append(np.arange(0,lon_obs_sub.shape[0],3),[lon_obs_sub.shape[0]-1]):
            for i_lon in np.append(np.arange(0,lon_obs_sub.shape[1],3),[lon_obs_sub.shape[1]-1]):
                angles[j,i_lat,i_lon] = sun_position(DJF_dates_all[j],
                                          lon_obs_sub[i_lat,i_lon],
                                          lat_obs_sub[i_lat,i_lon])

        elevation_angle[j] = angles[j,:,:,0]
        if azimuth_angle_calc:
            azimuth_angle[j] = angles[j,:,:,1]
            azimuth_angle[j] = fill_nans(azimuth_angle[j])
        elevation_angle[j] = fill_nans(elevation_angle[j])
    
    for i in np.arange(lon_obs_sub.shape[1]):
        # Spatial interpolation
        print i
        elevation_angle[:,:,i] = fill_nans(elevation_angle[:,:,i])
        if azimuth_angle_calc:
            azimuth_angle[:,:,i] = fill_nans(azimuth_angle[:,:,i])
    
    if azimuth_angle_calc:
        return elevation_angle, azimuth_angle
    else:
        return elevation_angle

def rad2grad(rad):
    return 180/np.pi*rad

def grad2rad(grad):
    return np.pi/180*grad


def fill_nans(indata, method='linear'):
    """
    Fills a matrix by linear interpolation.
    Fill NaN values in the input array `indata`.
    """
    # Find the non-NaN indices
    inds = np.nonzero(~np.isnan(indata))
    # Create an `out_inds` array that contains all of the indices of indata.
    out_inds = np.mgrid[[slice(s) for s in indata.shape]].reshape(indata.ndim, -1).T
    # Perform the interpolation of the non-NaN values to all the indices in the array:
    return griddata(inds, indata[inds], out_inds, method=method).reshape(indata.shape)



    
def sun_earth_distance(d):
    '''
    Claculate sun earth distance for a given date
    
    Parameters
    ----------
    d : datetime.date or datetime.datetime
    
    Returns
    -------
    sun_earth_distance: float in Astronomical Units, 1AU = mean E-S distance    

    
    '''
    ###################################################
    # CREATE SUN OBJECT                               #
    ###################################################
    sun = ephem.Sun(d)

    ###################################################
    # CREATE OBSERVER OBJECT                          #
    ###################################################
    site = ephem.Observer()
    #site.lon = str(lon)
    #site.lat = str(lat)
    site.date = d

    sun.compute(site)
    sun_earth_distance = sun.earth_distance #radiant
    return sun_earth_distance
    


if __name__ == "__main__":
    d = dt.datetime.now()
    s = datetime_to_seconds(d)
    x = DateTime.from_seconds(s)
    y = x.to_seconds()
#    d = DateTime.now()



def ist_schaltjahr(jahr):
    """
    liefert True, wenn das gegebene Jahr ein Schaltjahr ist.
    Seit Ende 1582 gilt der Gregorianische Kalender, bei der Bestimmung 
    sind folgende Regeln anzuwenden:
        In allen Jahren, deren Jahreszahl durch vier teilbar ist, ist der 
        29. Februar ein Schalttag und damit ist dieses Jahr ein Schaltjahr.
        Eine Ausnahme bilden allerdings die vollen Jahrhundertjahre 1700, 
        1800, 1900 usw., auch Säkularjahre genannt. Hiervon erhalten nur 
        diejenigen einen Schalttag, deren Jahreszahl durch 400 teilbar ist. 
        Jedes vierte Säkularjahr ist somit ein Schaltjahr.
    Für alle Jahre <= 1582 liefert die Funktion False, weil da das
    Schaltjahr nicht definiert ist, sonst gilt obige Regel.
    """
    if jahr <= 1582:
        schaltjahr = False
    elif jahr % 400 == 0:
        schaltjahr = True
    elif jahr % 100 == 0:
        schaltjahr = False
    elif jahr % 4 == 0:
        schaltjahr = True
    else:
        schaltjahr = False
    return schaltjahr


def index_in_datetime_vec(datum_all, season='', ):


    if season == 'spring':
        s_month = np.array([3,4,5]) 
    elif season == 'summer':
        s_month = np.array([6,7,8])
    elif season == 'autumn':
        s_month = np.array([9,10,11])
    elif season == 'winter':
        s_month = np.array([12,1,2])

    pd_datum_all = pd.DatetimeIndex(datum_all)
    pd_years = set(np.sort(pd_datum_all.year))
    

    # Create datetime vector for spring,summer,autumn
    if season != 'winter':
        for i_year, year_i in enumerate(pd_years):
            start_t = datetime.datetime(int(year_i),s_month[0],1,13)
            days_in_season = ( monthrange(int(year_i), s_month[0])[1] + 
                               monthrange(int(year_i), s_month[1])[1] + 
                               monthrange(int(year_i), s_month[2])[1] )
            _season_dates = np.array([start_t +dt.timedelta(days=i) for i in xrange(days_in_season)])
            if i_year == 0:
                season_dates=array(_season_dates)
            else:
                season_dates = array(list(season_dates) + list(_season_dates))
    
        # find spring corresponding index in datum_all
        i_season = [index for index,date in enumerate(datum_all) if date in season_dates]
    
    elif season == 'winter':
        s_month = np.array([12,1,2])
        for i_year, year_i in enumerate(pd_years):
            start_t = datetime.datetime(int(year_i),1,01,13)
            days_in_winter = ( monthrange(int(year_i), 1)[1] + 
                               monthrange(int(year_i), 2)[1])
            _winter_dates = np.array([start_t +dt.timedelta(days=i) for i in xrange(days_in_winter)])
            if i_year == 0:
                winter_dates=array(_winter_dates)
            else:
                winter_dates = array(list(winter_dates) + list(_winter_dates))
                
        i_winter = [index for index,date in enumerate(datum_all) if date in winter_dates]

        for i_year, year_i in enumerate(pd_years):
            start_t = datetime.datetime(int(year_i), 1,1,13)
            days_in_winter = monthrange(int(year_i), 12)[1]
            winter_dates = np.array([start_t +dt.timedelta(days=i) for i in xrange(days_in_winter)])
            i_winter.extend([index for index,date in enumerate(datum_all) if date in winter_dates])
        
        i_season = i_winter
    return i_season




def index_in_datetime_vec_v2(datum_all, season='', ):

    if season == 'spring':
        s_month = np.array([3,4,5]) 
    elif season == 'summer':
        s_month = np.array([6,7,8])
    elif season == 'autumn':
        s_month = np.array([9,10,11])
    elif season == 'winter':
        s_month = np.array([12,1,2])

    pd_datum_all = pd.DatetimeIndex(datum_all)
    pd_years = set(np.sort(pd_datum_all.year))
    

    # Create datetime vector for spring,summer,autumn
    if season != 'winter':
        for i_year, year_i in enumerate(pd_years):
            pddf_datum_all = pd.DataFrame(np.arange(len(datum_all)),index=datum_all,columns=['col1'])
            str1 = str(year_i) + '-%02d'  %s_month[0]
            str2 = str(year_i) + '-%02d'  %s_month[-1]
            _season_dates = pddf_datum_all.col1.loc[str1:str2]

            if i_year == 0:
                season_dates= np.array(_season_dates)
            else:
                season_dates = np.array(list(season_dates) + list(_season_dates))
        i_season = season_dates

    elif season == 'winter':
        s_month = np.array([12,1,2])
        for i_year, year_i in enumerate(pd_years):
            pddf_datum_all = pd.DataFrame(np.arange(len(datum_all)),index=datum_all,columns=['col1'])
            str1 = str(year_i) + '-%02d'  %s_month[1]
            str2 = str(year_i) + '-%02d'  %s_month[-1]
            _winter_dates = pddf_datum_all.col1.loc[str1:str2]
            if i_year == 0:
                winter_dates = np.array(_winter_dates)
            else:
                winter_dates = np.array(list(winter_dates) + list(_winter_dates))
                
        i_winter = np.array(winter_dates)

        for i_year, year_i in enumerate(pd_years):
            str1 = str(year_i) + '-%02d'  %s_month[0]
            _winter_dates = pddf_datum_all.col1.loc[str1]
            i_winter = np.append(i_winter, _winter_dates)
        i_season = i_winter
    return i_season


def index_in_datetime_vec_month(datum_all, month ):

    pd_datum_all = pd.DatetimeIndex(datum_all)
    pd_years = set(np.sort(pd_datum_all.year))
    

    # Create datetime vector for spring,summer,autumn

    for i_year, year_i in enumerate(pd_years):
        pddf_datum_all = pd.DataFrame(np.arange(len(datum_all)),index=datum_all,columns=['col1'])
        str1 = str(year_i) + '-%02d'  %month
        _season_dates = pddf_datum_all.col1.loc[str1]

        if i_year == 0:
            season_dates= np.array(_season_dates)
        else:
            season_dates = np.array(list(season_dates) + list(_season_dates))
    i_season = season_dates

    return i_season

def index_in_datetime_vec_year(datum_all, year):
    pd_datum_all = pd.DatetimeIndex(datum_all)
    pd_years = set(np.sort(pd_datum_all.year))
    
    pddf_datum_all = pd.DataFrame(np.arange(len(datum_all)),index=datum_all,columns=['col1'])
    str1 = str(year)
    _season_dates = pddf_datum_all.col1.loc[str1]
    return np.array(_season_dates)

def index_in_datetime_vec_1month_1year(datum_all,year,month):
    month_index = index_in_datetime_vec_month(datum_all, month)
    year_index = index_in_datetime_vec_year(datum_all,year)
    #month_in_year_index = list(set(month_index) & set(year_index))
    month_in_year_index = np.sort(list(set(year_index).intersection(month_index)))
    return month_in_year_index

def index_in_datetime_vec_1month_1year_1day(datum_all,year,month,day):
    #pd_datum_all = pd.DatetimeIndex(datum_all)
    pddf_datum_all = pd.DataFrame(np.arange(len(datum_all)),index=datum_all,columns=['col1'])
    str1 = str(year) + '-%02d-%02d'  % (month,day)
    i_dates = pddf_datum_all.col1.loc[str1]
    return i_dates

def array_of_dates(years, time_range, time_range_min=[00]):
    '''
    Defines an array with datetime objects of given years, hours and minutes 
    in sorted order
    
    Parameters
    ----------
    years : list
            Contains years of interest
    time_range: list
                Contains hours of interest
    time_range_min: list, optional
                    Contains minutes of interest
    Returns
    -------
    Datetime array
    
    
    '''
    n_year_days_list = np.nan * np.empty( (len(years)) )
    # define date_all vector
    for time_range_min_i in time_range_min:
        for time in time_range:
            for i_year,year_i in enumerate(years):
                n_year_days = sum( [ (calendar.monthrange(year_i,month_i)[1]) 
    				  for month_i in range(1,13) ] )
                n_year_days_list[i_year] = n_year_days
                start_t = dt.datetime(year_i,01,01,time)
                datum = np.array([start_t + dt.timedelta(days=i)
    			      + dt.timedelta(minutes=time_range_min_i) for 
    			      i in xrange(n_year_days)])
                if i_year == 0:
                    datum_all = np.array(datum)
                    assert i_year==0
                else:
                    datum_all = np.array(list(datum_all)+list(datum))
            if time == time_range[0]:
                datum_all1 = datum_all
            else:
                datum_all1 = np.array(list(datum_all1) + list(datum_all))
        if time_range_min_i == time_range_min[0]:
            datum_all2 = datum_all1
        else:
            datum_all2 = np.array(list(datum_all2) + list(datum_all1))
    return np.array(sorted(datum_all2))



def match_dates(date1, date2):
    '''
    Findes indexes of common date
    
    INPUT
    -----
    date1: np.array of datetime objects
    date2: np.array of datetime objects
    
    OUTPUT
    ------
    ind_in_date1: np.array of integers
    ind_in_date2: np.array of integers
    '''
    date1_s = np.array(datetime_to_seconds(date1)).astype(int) #+ 5*60
    date2_s = np.array(datetime_to_seconds(date2)).astype(int)
    ind_in_date1 = []
    ind_in_date2 = []
    #for i,date1_i in enumerate(date1_s[0:-1]):
    for i,date1_i in enumerate(date1_s):
        if date1_i in date2_s:
            ind = np.where(date1_i == date2_s)[0]
            N_hits = len(ind)
            if len(ind) == 1:
                ind_in_date2.append(ind[0])
                ind_in_date1.append(i)
    ind_in_date1 = np.array(ind_in_date1).astype(int)
    ind_in_date2 = np.array(ind_in_date2).astype(int)
    return ind_in_date1, ind_in_date2


def get_indexes_of_time_span(date_vec_in, start_time, end_time):
    '''
    Findes indexes of given time span within every day
    
    INPUT
    ----
    date_vec_in : vector of datetime objects
    start_time: string of beginning time 
                example: '11:15'
    end_time: string of ending time
              example: '14:00'

    OUTPUT
    ------
    indexes: indexes of interest
    '''
    pddf_date_in = pd.DataFrame(np.arange(len(date_vec_in)),index=date_vec_in,columns=['col1'])
    return np.array(pddf_date_in.between_time(start_time,end_time))[:,0]



