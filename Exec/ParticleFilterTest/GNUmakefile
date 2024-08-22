# AMREX_HOME defines the directory in which we will find all the AMReX code
AMREX_HOME ?= ../../subprojects/amrex

# TOP defines the directory in which we will find Source, Exec, etc
TOP = ../..

# compilation options
COMP    = gnu
USE_MPI = TRUE
# Use with Async IO
MPI_THREAD_MULTIPLE = FALSE
USE_OMP = FALSE
USE_CUDA = FALSE

USE_SUNDIALS      = TRUE
USE_FORT_ODE = FALSE
SUNDIALS_ROOT = /global/cfs/cdirs/m4106/jmsexton/sundials_shared/sundials/instdir
#SUNDIALS_ROOT ?=  ${CURDIR}/$(TOP)/../sundials/instdir
USE_HDF5 = TRUE
HDF5_HOME = /opt/cray/pe/hdf5/1.12.2.9

PROFILE       = FALSE
TRACE_PROFILE = FALSE
COMM_PROFILE  = FALSE
TINY_PROFILE  = FALSE

PRECISION = DOUBLE
USE_SINGLE_PRECISION_PARTICLES = TRUE
DEBUG     = FALSE

# physics
DIM      = 3
USE_HEATCOOL = TRUE
USE_SAVE_REACT = FALSE
USE_AGN = FALSE
ifeq ($(NO_HYDRO),TRUE)
USE_SDC = FALSE
USE_SUNDIALS = FALSE
USE_FUSED = FALSE
else
ifeq ($(USE_HEATCOOL),TRUE)
USE_SDC = TRUE
USE_SUNDIALS = TRUE
ifeq ($(USE_HIP),TRUE)
USE_FUSED ?= $(USE_HIP)
endif
USE_FUSED ?= $(USE_CUDA)
else
USE_SDC = FALSE
USE_SUNDIALS = FALSE
USE_FUSED = FALSE
endif
endif

USE_SDC = FALSE
USE_FUSED = TRUE
USE_HEATCOOL=FALSE

USE_CONST_SPECIES = TRUE
NEUTRINO_PARTICLES = FALSE
NEUTRINO_DARK_PARTICLES = FALSE

USE_OWN_BCS = FALSE

# Halo finder
BOOST_INLUDE_DIR := ${OLCF_BOOST_ROOT}/include/boost
REEBER = FALSE

Bpack := ./Make.package
Blocs := .

include $(TOP)/Exec/Make.Nyx
