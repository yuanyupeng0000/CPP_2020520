#******************************************************************************
#
# Makefile - Rules for building the board finder utility.
#
# Copyright (c) 2009 Texas Instruments Incorporated. All rights reserved.
# Software License Agreement
# 
# Texas Instruments (TI) is supplying this software for use solely and
# exclusively on TI's microcontroller products. The software is owned by
# TI and/or its suppliers, and is protected under applicable copyright
# laws. You may not combine this software with "viral" open-source
# software in order to form a larger program.
# 
# THIS SOFTWARE IS PROVIDED "AS IS" AND WITH ALL FAULTS.
# NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT
# NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE APPLY TO THIS SOFTWARE. TI SHALL NOT, UNDER ANY
# CIRCUMSTANCES, BE LIABLE FOR SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
# DAMAGES, FOR ANY REASON WHATSOEVER.
# 
# This is part of revision 5450 of the Stellaris Firmware Development Package.
#
#******************************************************************************

#
# The name of this application.
#
APP:=finder

#
# The object files that comprise this application.
#
OBJS:=finder.o                 \
      gui.o                    \
      fltk-1.1.9/lib/libfltk.a

#
# The resources for the Windows version.
#
RES:=finder.res

#
# Include the generic rules.
#
include ../toolsdefs

#
# Modify some of the generic defines.
#
LD:=${CXX}
CFLAGS:=${CFLAGS} -g -I fltk-1.1.9
LDFLAGS:=${LDFLAGS} -g
ifneq ($(findstring CYGWIN, ${os}), )
LDFLAGS:=${LDFLAGS} -mwindows
LIBS:=ole32 uuid wsock32 iphlpapi
else
LIBS:=X11 Xpm pthread
endif

#
# Fetch the tarball that contains FLTK.
#
fltk-1.1.9-source.tar.bz2:
	@URL=http://ftp2.easysw.com/pub/fltk/1.1.9/fltk-1.1.9-source.tar.bz2; \
	 if [ x${VERBOSE} = x ];                                              \
	 then                                                                 \
	     echo "  WGET  `basename $${URL}`";                               \
	     wget $${URL} > /dev/null 2>&1;                                   \
	 else                                                                 \
	     echo wget $${URL};                                               \
	     wget $${URL};                                                    \
	 fi

#
# Unpack and build FLTK.
#
fltk-1.1.9/lib/libfltk.a: fltk-1.1.9-source.tar.bz2
	@if [ ! -d fltk-1.1.9 ];                        \
	 then                                           \
	     if [ x${VERBOSE} = x ];                    \
	     then                                       \
	         echo "  TAR   ${^}";                   \
	     else                                       \
	         echo tar -xjf ${^};                    \
	     fi;                                        \
	     tar -xjf fltk-1.1.9-source.tar.bz2;        \
	 fi;                                            \
	 cd fltk-1.1.9;                                 \
	 if [ ! -f config.h ];                          \
	 then                                           \
	     if [ x${VERBOSE} = x ];                    \
	     then                                       \
	         echo "  CONFIGURE";                    \
	     else                                       \
	         echo ./configure;                      \
	     fi;                                        \
	     ./configure --enable-localjpeg             \
	                 --enable-localzlib             \
	                 --enable-local-png             \
	                 --enable-threads               \
	                 --disable-gl                   \
	                 CFLAGS="-g0 -O2"               \
	                 CPPFLAGS="-g0 -O2"             \
	                 CXXFLAGS="-g0 -O2" > log 2>&1; \
	 fi;                                            \
	 if [ x${VERBOSE} = x ];                        \
	 then                                           \
	     echo "  MAKE";                             \
	 else                                           \
	     echo make;                                 \
	 fi;                                            \
	 make >> log 2>&1

#
# Clean out additional build products.
#
clean::
	@rm -rf fltk-1.1.9 finder.res gui.cxx gui.h

#
# A dependency to ensure that fltk is built before the application.
#
finder.o: fltk-1.1.9/lib/libfltk.a gui.cxx

#
# A rule to generate gui.cxx.
#
gui.cxx: gui.fl fltk-1.1.9/lib/libfltk.a
	@if [ x${VERBOSE} = x ];          \
	 then                             \
	     echo "  FLUID gui.fl";       \
	 else                             \
	     echo "fluid -c gui.fl";      \
	 fi;                              \
	 fltk-1.1.9/fluid/fluid -c gui.fl

#
# A dependency to ensure that the Windows resources are rebuilt if the icon
# file is modified.
#
finder.res: finder.ico
