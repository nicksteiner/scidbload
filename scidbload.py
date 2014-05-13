#!/usr/bin/python

'''
Simple Scidb Loader for Numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple library to load SciDB arrays via the python api.

2013, revised 2014


Author: Nick Steiner <nick.steiner@gmail.com>
Orcid: 0000-0001-5943-8400
See http://github.com/


The MIT License (MIT)
~~~~~~~~~~~~~~~~~~~~~

Copyright (c) [2014] [Nicholas Steiner]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


'''

from __future__ import print_function, division, absolute_import

import os
import sys
import ipdb

import numpy as np


# requires scidb api and scidb
__scidbver__ = os.getenv('SCIDB_VER', '13.12')
sys.path.append('/opt/scidb/{}/lib'.format(__scidbver__))
import scidbapi

'''
Constants
~~~~~~~~~
'''

_np_typename = lambda s: np.dtype(s).descr[0][1]

SDB_NP_TYPE_MAP = {'bool': _np_typename('bool'),
                   'float': _np_typename('float32'),
                   'double': _np_typename('float64'),
                   'int8': _np_typename('int8'),
                   'int16': _np_typename('int16'),
                   'int32': _np_typename('int32'),
                   'int64': _np_typename('int64'),
                   'uint8': _np_typename('uint8'),
                   'uint16': _np_typename('uint16'),
                   'uint32': _np_typename('uint32'),
                   'uint64': _np_typename('uint64'),
                   'char': _np_typename('c'),
                   'datetime': _np_typename('datetime64[s]'),
                   'string': _np_typename('object')}

IGNORE = (scidbapi.swig.ConstChunkIterator.IGNORE_OVERLAPS) | \
    (scidbapi.swig.ConstChunkIterator.IGNORE_EMPTY_CELLS)

'''
Classes
~~~~~~~
'''

class Scidb:
    ''' Scidb handles scidb connections.'''


    _name = 'Scidb'

    def __init__(self, host='localhost', port=1239):
        self.host = host
        self.port = port
    
    def __enter__(self):
        self.scidb = scidbapi.connect(self.host, self.port)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
         self.scidb.disconnect()

    def _scan_array(self, sdb_name):
        return self.scidb.executeQuery('save({}, "\dev\null")'.format(sdb_name)).array

    def _query(self, afl):
        return self.scidb.executeQuery(afl).array


class ScidbArray(object):
    ''' Scidb array class.'''

    #afl_fun =  "save({}, '\dev\null')".format
    afl_fun = 'scan({})'.format
    def __str__(self):
        return self.description.__str__()

    def __repr__(self):
        return '<ScidbArray({})>'.format(self.__str__())

    def __init__(self, array_name, verbose=True, host='localhost', port=1239):

        self.host = host
        self.port = port
        self.afl = self.afl_fun(array_name)

        with Scidb(self.host, self.port) as db:
            self.array = db._query(self.afl)
        #self.array = swig_array
        #ipdb.set_trace()
        self.description = self.array.getArrayDesc()
        self.attributes = self.description.attributes
        self.dimensions = self.description.dimensions
        self.attsLoaded = []
        if verbose:
            atts_ = '\n'.join([att.Name for att in self.description.attributes])
            print('\nAttributes:\n {0}'.format(atts_))

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, val):
        self._attributes = dict([v.Name, v] for v in val)

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, val):
        self._dimensions = val

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = Description(val)

    @property
    def data(self):
        if not hasattr(self, '_data'):
            self._data = dict()
            for key, val in self.attributes.iteritems():
                iter_ = self.array.getConstIterator(val.Id)
                self._data[key] = Data(val, self.dimensions, iter_)
        return self._data

    @property
    def sa(self):
        ''' Numpy structured array. '''
        if not hasattr(self, '_sa'):
            n_, t_, a_ = zip(*[(n, d.array.dtype, d.array) for n, d 
                    in self.data.iteritems()])
            dtype = dict(names=n_, formats=t_)
            self._sa = np.empty(shape=a_[0].shape, dtype=dtype)
            for n, a in zip(n_,a_):
                self._sa[n] = a
        return self._sa
    
    @property
    def sp(self):
        ''' Numpy sparse-structured array. '''
        if not hasattr(self, '_sa'):
            n_, t_, a_ = zip(*[(n, d.array.dtype, d.array) for n, d 
                    in self.data.iteritems()])
            dtype = dict(names=n_, formats=t_)
            self._sa = np.empty(shape=a_[0].shape, dtype=dtype)
            for n, a in zip(n_,a_):
                self._sa[n] = a
        return self._sa

    def load_sparse(self):
        pass

    def load_all(self, type_='dense'):
        if type_ == 'dense':
            self.load(self.attributes.keys())
        elif type_ == 'sparse':
            self.load_sparse(self.attributes.keys())
        return self.data

    def load(self, att_list):
        assert len(att_list) > 0
        with Scidb(self.host, self.port) as db:
            self.array = db._query(self.afl)
            data_list = [self.data[att] for att in att_list]
            ##TODO optimize here
            while not data_list[0].iterator.end():
                while not data_list[0].chunk.iter.end():
                    pos_ = [dim.relative_position(data_list[0]. \
                        chunk.iter.getPosition()[i]) for i, dim in 
                        enumerate(self.dimensions)]
                    for data in data_list:
                        value = data.chunk.iter.getItem()
                        if not value.isNull():
                            data.array[tuple(pos_)] = scidbapi.getTypedValue(value, data.att.Type)
                        data.chunk.iter.increment_to_next()
                for data in data_list:
                    data.iterator.increment_to_next()
                    try:
                        data.set_chunk()
                    except:
                        continue
        self.attsLoaded.extend(att_list)

    def __call__(self):
        # set properties
        self.load_all()
        return self

class Data(object):
    '''Numpy data array '''
    def __init__(self, att, dims, iter_, null=np.NAN):
        self.att = att
        self.dims = dims
        self.null = null
        self.iterator = iter_
        #self.chunk = iter_.getChunk()

    def __repr__(self):
        return 'DATA({0})'.format(self.att.Name)

    @property
    def array(self):
        if not hasattr(self, '_array'):
            self._array = self._get_array()
        return self._array

    def _get_array(self):
        size_ = tuple([abs(dim.Start - dim.EndMax) for dim in self.dims])
        data = np.empty(size_, dtype=SDB_NP_TYPE_MAP[self.att.Type])
        try:
            data[:] = self.null
        except:
            data[:] = -9999
        return data

    def set_chunk(self):
        self.chunk = self.iterator.getChunk()

    @property
    def chunk(self):
        if not hasattr(self, '_chunk'):
            self.set_chunk()
        return self._chunk

    @chunk.setter
    def chunk(self, swig_chunk):
        self._chunk = Chunk(swig_chunk, self.att.Type)


class Chunk(object):


    def __init__(self, swig_chunk, type_):
        self.chunk = swig_chunk
        self.iter = swig_chunk.getConstIterator(IGNORE)
        self.type = type_


class Description:


    _name = 'description'

    def __str__(self):

        return '{0} {1}'.format(self.Name, self.anonymous_schema)

    def __init__(self, description):
        self.description = description
        self.Name = description.getName().split('@')[0]
        self.NumberOfChunks = description.getNumberOfChunks()
        #self.Comment = description.getComment()
        self.CurrSize = description.getCurrSize()
        self.PartitioningSchema = description.getPartitioningSchema()
        self.Size = description.getSize()
        self.UAId = description.getUAId()
        #self.Flags = description.getFlags()
        self.UsedSpace = description.getUsedSpace()
        self.Id = description.getId()
        self.VersionId = description.getVersionId()
        
    @property
    def anonymous_schema(self):
        return '<{}> [{}]'.format(self.att_str, self.dim_str)

    @property
    def dim_str(self):
        return ','.join([dim.__str__() for dim in self.dimensions])

    @property
    def att_str(self):
        return ','.join([att.__str__() for att in self.attributes])

    @property
    def attributes(self):
        if not hasattr(self, '_attributes'):
            self._attributes = []
            atts = self.description.getAttributes()
            for i in range(atts.size()):
                att = Attribute(atts[i])
                if att.Name not in ['EmptyTag']:
                    self._attributes.append(att)
        return self._attributes

    @property
    def dimensions(self):
        if not hasattr(self, '_dimensions'):
            self._dimensions = []
            dims = self.description.getDimensions()
            for i in range(dims.size()):
                dim = Dimension(dims[i])
                self._dimensions.append(dim)
        return self._dimensions


class Attribute:
    '''Scidb array attribute.'''

    _name = 'attribute'

    def __str__(self):
        type_ = self.Type
        if self.IsNullable:
            type_ += ' null'
        return "{}:{}".format(self.Name, type_)

    def __repr__(self):
            return "<Attribute({0})>".format(self.__str__())
    def __init__(self, attribute):
        self.attribute = attribute
        #self.Comment = attribute.getComment()
        self.DefaultCompressionMethod = attribute.getDefaultCompressionMethod()
        self.DefaultValueExpr = attribute.getDefaultValueExpr()
        #self.Flags = attribute.getFlags()
        self.Id = attribute.getId()
        self.Name = attribute.getName()
        self.Reserve = attribute.getReserve()
        self.Size = attribute.getSize()
        self.Type = attribute.getType()
        self.VarSize = attribute.getVarSize()
        self.IsNullable = attribute.isNullable()

class Dimension:
    '''Scidb array dimension.'''


    _name = 'dimension'

    def __str__(self):
        return "{BaseName}={StartMin}:{EndMax},{ChunkInterval},{ChunkOverlap}"\
            .format(**self.__dict__)

    def __repr__(self):
        return "<Dimension({0})>".format(self.__str__())

    def __init__(self, dimension):
        self.dimension = dimension
        self.BaseName = dimension.getBaseName()
        self.ChunkInterval = dimension.getChunkInterval()
        self.ChunkOverlap = dimension.getChunkOverlap()
        #self.Comment = dimension.getComment()
        self.CurrEnd = dimension.getCurrEnd()
        self.CurrLength = dimension.getCurrLength()
        self.CurrStart = dimension.getCurrStart()
        self.EndMax = dimension.getEndMax()
        #self.Flags = dimension.getFlags()
        #self.FuncMapOffset = dimension.getFuncMapOffset()
        #self.FuncMapScale = dimension.getFuncMapScale()
        self.Length = dimension.getLength()
        self.HighBoundary = dimension.getHighBoundary()
        self.Start = dimension.getStart()
        self.LowBoundary = dimension.getLowBoundary()
        self.NamesAndAliases = dimension.getNamesAndAliases()
        self.Start = dimension.getStart()
        self.StartMin = dimension.getStartMin()
        #self.Type = dimension.getType()

    def relative_position(self, index):
        return index - self.LowBoundary - 1


class List(ScidbArray):
    ''' List of current scidb arrays.'''


    afl_fun = str

    def __init__(self, host='localhost'):
        #super(ScidbArray, self).__init__('list()')
        ScidbArray.__init__(self, 'list()', verbose=False, host=host)
        self.load_all()

    def __call__(self):
        # Updateable
        self.load_all()

    @property
    def name(self):
        return self.data['name']

    @property
    def name_list(self):
        return self.name.array

    @property
    def id(self):
        return self.data['id']

    @property
    def schema(self):
        return self.data['schema']

    @property
    def schema_lookup(self):
        if not hasattr(self, '_schema_lookup'):
            self._schema_lookup = dict(zip(self.name_list, self.schema.array))
        return self._schema_lookup


class Query(ScidbArray):
    '''Submit AFL query to scidb.
        (AQL not supported yet ...)

    Arguments:
    afl_string:     Scidb command in AFL format
    '''

    afl_fun = str

