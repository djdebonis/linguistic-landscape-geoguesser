# geo_utils.py
import regex as re

def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + minutes / 60 + seconds / 3600

    if direction.upper() in ["S", "W"]:
        decimal *= -1

    return decimal

def parse_dms_coordinate(coord_string):
    """
    Parses a coordinate string like:
    39°44'24.7"N 104°51'14.8"W

    Returns:
        (latitude, longitude)
    """

    pattern = r"""(
        \d+)[°]\s*
        (\d+)'\s*
        ([\d\.]+)"\s*
        ([NSEW])
    """

    matches = re.findall(pattern, coord_string, re.VERBOSE)

    if len(matches) != 2:
        raise ValueError("Could not parse coordinate string")

    lat_match = matches[0]
    lon_match = matches[1]

    lat = dms_to_decimal(
        float(lat_match[0]),
        float(lat_match[1]),
        float(lat_match[2]),
        lat_match[3]
    )

    lon = dms_to_decimal(
        float(lon_match[0]),
        float(lon_match[1]),
        float(lon_match[2]),
        lon_match[3]
    )

    return lat, lon


def greet():
    print("hello from geo_utils")