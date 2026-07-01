#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from configparser import ConfigParser


# Re-write configure class, enable to distinguish betwwen upper and lower letters
class myconf(ConfigParser):
    def __init__(self,defaults=None):
        ConfigParser.__init__(self, defaults=None)
    def optionxform(self, optionstr):
        return optionstr
