
import warnings
from math import sqrt
from typing import List, Union

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.contour import ContourSet
from matplotlib.image import AxesImage


class SubPlot():
    def __init__(self, ncols, nrows, name, constrained_layout=True) -> None:
        self.ncols = ncols
        self.nrows = nrows
        self.fig = plt.figure(figsize=[6.4*ncols, 4.8*nrows], constrained_layout=constrained_layout) #constrained_layout=True
        self.gs = self.fig.add_gridspec(nrows, ncols)
        self.size = nrows*ncols*2+10
        if name: self.fig.suptitle(name,size=self.size,y=1+self.size/1000)
    def ax(self,x,y) -> Axes:
        return self.fig.add_subplot(self.gs[x,y])
    def addrow(self):
        pass


# TODO: subplot
# - sharex, sharey
# - design
# - kwargs
# - wspace, hspacce
# - add empty Axeses if colname or rowname is True


class megaplot():

    def __init__(self, *argv, fig=None, gs=None, nrows=None, ncols=None, title=None, xlabel=None, ylabel=None, constrained_layout=False, log=True, **kwargs):
        """
        A class used to make matplotlib subplot and its changes more convenient.

        Args:
            *argv: Tuple of (nrows, ncols) if provided as separate arguments.
            fig (pyplot.figure): The figure to use. If not provided, a new figure will be created.
            gs (GridSpec): The GridSpec to use. If not provided, a new GridSpec will be created.
            nrows (int): Number of rows in the subplot grid. If not provided, it will be determined based on the fig or gs.
            ncols (int): Number of columns in the subplot grid. If not provided, it will be determined based on the fig or gs.
            title (str): The title of the figure.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            constrained_layout (bool): Whether to use constrained layout for the figure.
            log (bool): Whether to print logs.
            **kwargs: Additional keyword arguments to be passed to the figure creation.

        Attributes:
            nrows (int): Number of rows in the subplot grid. Do not change directly!
            ncols (int): Number of columns in the subplot grid. Do not change directly!
            log (bool): Whether to print logs.
            row_names (list of str): Names of the rows.
            col_names (list of str): Names of the columns.
            fig (pyplot.figure): The figure.
            table (numpy.array): The table containing the indices of the fig.axes.

        Methods:
            expand(nrows, ncols): Increases the number of rows and/or columns in the subplot grid.
            transpose(): Transpose the subplot grid.
            ax(x, y): Get a specific axes at the given position (x, y).
            rownames(positions, names): Set names for some of the rows.
            colnames(positions, names): Set names for some of the columns.
            show(): Show the figure.
            savefig(filename, type='png', path='outputs', makedir=False, log=None, **kwargs): Save the figure.
            append2pdf(pdf): Save the figure to an existing PDF.

        Example:
            ```# Create 2x2 figure and two Axes at [0,:] and [1,0]
            myfig = subplot(2,2,constrained_layout=True,title='title',xlabel='axis')
            ax = myfig[0,:]
            ax.plot([2,1,3])
            ax = myfig[1,0]
            ax.plot([2,1,3])

            # myfig.show()

            # Expand figure size to 3x3 (rows=3, cols+=1)
            myfig.expand(3,myfig.ncols+1)

            # Create a new Axes in the new figure named 'new Axes'
            ax = myfig[:,2]
            ax.plot([3,1,2])
            ax.set_title('new Axes')

            # Print myfig.table in the terminal
            print(myfig.table)
            # myfig.show()

            # Modify Axes at [1,0]
            ax = myfig[1,0]
            ax.plot([3,3,2])
            ax.set_title('modified Axes')

            # Create a new Axes ([0,1]) in an alternative mode on top of an existing (our first Axes)
            ax = myfig.ax(0,1)
            ax.plot([10,2])
            ax.set_title('displaced Axes')

            # Create titles for the rows and the columns
            myfig.rownames([0,1,2],['a','b','new row'])
            myfig.colnames([0,1,2],['a','b','new column'])

            # Transpose the whole figure (change columns and rows)
            myfig.transpose()

            # Show our figure
            myfig.show()

            # Save our figure
            myfig.savefig('test',type='pdf',path='test')```

        """
        if len(argv)==2:
            nrows=argv[0]
            ncols=argv[1]

        if fig != None:

            # Set figure
            self.fig = fig

            # Set nrows, nfigures
            # nrows, ncols = self.gettable(nrows,ncols)

            # Set GridSpec, nrows, ncols
            if gs != None:
                self.gs = gs
                if not nrows: nrows = gs.nrows
                if not ncols: ncols = gs.ncols
            else: self.gs = self.fig.add_gridspec(nrows,ncols)

            # Resize figure
            self.expand(nrows, ncols)

        else:

            # Set figure
            self.fig = plt.figure(figsize=[6.4*ncols, 4.8*nrows], constrained_layout=constrained_layout)

            # Set list of geometry
            self.geometry = []
            self.table = np.zeros((nrows,ncols),dtype=np.int64)

            # Set GridSpec
            if gs != None: self.gs = gs
            elif nrows==0 or ncols==0: pass
            else: self.gs = self.fig.add_gridspec(nrows,ncols)

        # Update nrows and ncols
        self.nrows = nrows
        self.ncols = ncols
        self.log = log

        # Init objects
        # self.shareXaxis = self.getitem()
        self.suptexts = [None,None,None]
        if title:
            self.suptexts[0] = title
        if xlabel:
            self.suptexts[1] = xlabel
        if ylabel:
            self.suptexts[2] = ylabel
        self.row_names = [None for ___ in range(nrows)]
        self.col_names = [None for ___ in range(ncols)]

    def __getitem__(self, items) -> Axes:
        
        # Check if the values are lesser then self.nrows and self.ncols
        if abs(items[0]) >= self.nrows: raise ValueError(f'The first index ({items[0]}) is not smaller than nrows {self.nrows}')
        if abs(items[1]) >= self.ncols: raise ValueError(f'The second index ({items[1]}) is not smaller than ncols {self.ncols}')
            
        
        items = self.getCoordinates(items) # Transform items to concrete bounds
                
        if items in self.geometry:
            return self.get_axes(items)
        elif self.table.__getitem__(items).any() > 0:
            return self.get_multiple_axeses(items)
        else:
            return self.addaxes(items, log=False)  # only log if self.log == True
    
    def __setitem__(self, items, artist): # could this even work?
        # ax = self.fig.add_subplot(self.gs.__getitem__(items))
        # plt.sca(ax)
        print(artist, type(artist))
        print(artist.axes, type(artist.axes))
        
        ax = artist.axes
        
        self.fig.axes.append(ax)
        self.geometry.append(items)
        self.table = self.table_update(self.table, items, len(self.fig.axes))
        ax.set_subplotspec(self.gs.__getitem__(items))
    
    def __len__(self):
        return len(self.geometry)
    
    def __iter__(self):
        self.cnt = 0
        return self
    
    def __next__(self):
        if self.cnt < self.__len__():
            ret =  self.fig.axes[self.cnt]
            self.cnt += 1
            return ret
        else:
            raise StopIteration
    
    class area():
        def __init__(self, items:tuple):
            "items: tuple of two slices"
            self.x = items[0]
            self.y = items[1]
        def __iter__(self):
            self.cntx = self.x.start
            self.cnty = self.y.start
            self.stop = False
            return self
        def __next__(self):
            
            ret = (self.cntx,self.cnty)
            
            # If out of bounds, stop iteration
            if self.cnty == self.y.stop: raise StopIteration
            
            # Next tile
            self.cntx += 1
            if self.cntx == self.x.stop:
                self.cntx = self.x.start
                self.cnty += 1
            
            # Raise error if ran out
            if self.cntx > self.x.stop or self.cnty > self.y.stop: raise ValueError()
            
            return ret
    
    class getitem():
        def __init__(self,func):
            self.items = None
            self.func = func
        def __getitem__(self,items):
            self.items = items
            self.func
    
    def show(self) -> None:
        self.final()
        plt.show()
        self.reverse_final()
    
    def save(self, filename, type='png', path='outputs', makedir=False, log=None, **kwargs):
        """
        ### Params:
        filename: string of the file name
        type: 'png' or 'pdf'
        path: path to the file
        makedir: whether to make directories if not exist
        log (bool | None): whether to log
        """
        
        self.final()
        
        # Path
        if path[-1] != '/': slash = '/'
        else: slash = ''
        
        # Mkdir
        if makedir:
            import os
            os.makedirs(path,exist_ok=True)
        
        # Create and log pathnfile
        if '/' in filename: pathnfile = filename
        else: pathnfile = path+slash+filename
        if log or (log==None and self.log): print(f'Save {pathnfile}')
        
        if type == 'png':
            if not '.' in pathnfile: pathnfile = pathnfile + '.png'
            plt.savefig(pathnfile, **kwargs)

        if type == 'pdf':
            if not '.' in pathnfile: pathnfile = pathnfile + '.pdf'
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(pathnfile) as pdf: pdf.savefig(bbox_inches='tight')
        
        self.reverse_final()
    
    def append2pdf(self,pdf:PdfPages):
        """pdf: the handler of the file"""
        self.final()
        pdf.savefig(bbox_inches='tight')
        self.reverse_final()
    
    mathtextdict = {'{':'',
                    '}':'',
                    '_': ' '}
    
    def reverse_final(self):
        """
        Reverse the changes made by the final() method.
        """
        # Reset rows
        if self.ytitles:
            for cnt, name in enumerate(self.row_names):
                if name != None and self.table[cnt,0] > 0:
                    ax = self.fig.axes[self.table[cnt, 0]-1]
                    ax.set_ylabel(self.ytitles.pop(0))
        
        # Reset columns
        if self.xtitles:
            for cnt, name in enumerate(self.col_names):
                if name != None and self.table[cnt, 0] > 0:
                    ax = self.fig.axes[self.table[0, cnt]-1]
                    ax.set_title(self.xtitles.pop(0))
        
        # Reset fig attributes
        self.fig.suptitle('')
        self.fig.supxlabel('')
        self.fig.supylabel('')
        
        
    def final(self):
        
        self.ytitles = []
        self.xtitles = []
        
        # Rows
        for cnt, name in enumerate(self.row_names):
            if name != None and self.table[cnt,0] > 0:
                ax = self.fig.axes[self.table[cnt,0]-1]
                title = ax.get_ylabel()
                self.ytitles.append(title)
                name = str(name)
                name = ''.join(self.mathtextdict.get(c,c) for c in name) # delete chars, that corrupt the string in mathtex
                name = ' '.join(rf'$\bf{{{word}}}$' for word in str(name).split()) # or instead: name = ' '.join(rf'$**{word}**$' for word in str(name).split()) # bold by mathtex
                ax.set_ylabel(f'{name}\n{title}')
        
        # Cols
        for cnt, name in enumerate(self.col_names):
            if name != None and self.table[0,cnt] > 0:
                ax = self.fig.axes[self.table[0,cnt]-1]
                title = ax.get_title()
                self.xtitles.append(title)
                name = str(name)
                name = ''.join(self.mathtextdict.get(c,c) for c in name) # delete chars, that corrupt the string in mathtex
                name = ' '.join(rf'$\bf{{{word}}}$' for word in str(name).split()) # bold by mathtex
                ax.set_title(f'{name}\n{title}')
        
        # Suptexts
        if self.suptexts[0]:
            text = self.suptexts[0]
            text = str(text)
            text = ''.join(self.mathtextdict.get(c,c) for c in text) # delete chars, that corrupt the string in mathtex
            text = ' '.join(rf'$\bf{{{word}}}$' for word in str(text).split()) # bold by mathtex
            self.fig.suptitle(text,fontsize=self.textsize('suptitle'))
        if self.suptexts[1]:
            text = self.suptexts[1]
            self.fig.supxlabel(text,fontsize=self.textsize('supaxis'))
        if self.suptexts[2]:
            text = self.suptexts[2]
            self.fig.supylabel(text,fontsize=self.textsize('supaxis'))
    
    def get_axes(self,items) -> Axes:
        which_axes = self.geometry.index(items)
        return self.fig.axes[which_axes]
    
    def get_multiple_axeses(self, items): # check existing axeses in the specified area
        
        l = []
        
        for tile in self.area(self.getCoordinates(items)):
            
            if self.table.item(tile) > 0:
                l.append(self.fig.axes[int(self.table.item(tile))-1])
        
        return tuple(l)
    
    def expand(self, nrows, ncols):
        """
        https://github.com/matplotlib/matplotlib/issues/7225/
        
        None, if not to change
        """
        
        # Initialize
        if nrows == None: nrows = self.nrows
        if ncols == None: ncols = self.ncols
        oldrows, oldcols = self.nrows, self.ncols
        
        # Conditions:
        if nrows <= self.nrows and ncols <= self.ncols: return
        if nrows < self.nrows or ncols < self.ncols: raise ValueError('nrows and ncols must be equal or greater than original nrows and ncols')
        
        # Prepare gridspec & table & fig to the Axis rearrange
        gs = gridspec.GridSpec(nrows, ncols)
        if self.nrows==0 or self.ncols==0: # init if it was 0 so far
            # gs = self.fig.add_gridspec(nrows,ncols)
            gs.tight_layout(self.fig)
        table = np.zeros((nrows, ncols),dtype=np.int64)
        self.fig.set_size_inches(6.4*ncols, 4.8*nrows)
        # self.fig.set_figheight(self.nrows*4.8)
        # self.fig.set_figwidth(self.ncols*6.4)
        
        # Axes rearrange
        for i, ax, geo in zip(range(len(self.geometry)), self.fig.axes, self.geometry):
            pos = gs[geo].get_position(self.fig) # Bbox
            ax.set_position(pos)
            ax.set_subplotspec(gs[geo])
            table = self.table_update(table, geo, i+1)
        
        # Constrained layout
        if plt.rcParams['figure.constrained_layout.use'] == False:
            plt.rcParams['figure.constrained_layout.use'] = True
        
        self.nrows, self.ncols = nrows, ncols
        self.gs = gs
        self.table = table
        self.row_names = [___ for ___ in self.row_names] + [None for ___ in range(nrows - oldrows)]
        self.col_names = [___ for ___ in self.col_names] + [None for ___ in range(ncols - oldcols)]
       
    def table_update(self, table, items, nth_axes, log=True):
        """add items tuple to the table. warn if already occupied"""
        
        # Iterate through gridspec tiles of the designated area
        for tile in self.area(self.getCoordinates(items)):
            
            # Check if occupied
            if table.__getitem__(tile) == 1 and (log or self.log):
                warnings.warn(f'{items} is already in use') # occupied
            
            # Set table element to number of axes
            table.__setitem__(tile,int(nth_axes)) # self.table[tile[0],tile[1]] = 1
        
        return table
    
    def addaxes(self, items, log=True):
        
        self.geometry.append(self.getCoordinates(items))
        
        self.table = self.table_update(self.table, items, len(self.fig.axes)+1, log=log) # len(axes)+1 bcs u have not added yet the axes to the fig
            
        return self.fig.add_subplot(self.gs.__getitem__(items))
     
    def transformSizeToFitSubplots(self, sl, max:int): # Transorm slice to fit the subplots by nrwos and ncols
        if type(sl) == int: sl = slice(sl,sl+1)
        
        if type(sl) == slice:
            
            start, stop = sl.start, sl.stop
            
            if stop == None: stop = max
            if start == None: start = 0
            
            if stop < 0:
                stop += max
            if start < 0:
                start += max
                if stop < start: stop += max # is sl.stop == sl.start+1
            
            # Check
            if stop > max: raise ValueError(f'stop {stop} is greater than max {max}')
            
            return slice(start,stop)
        
        else: raise TypeError('I can only concretize only on int or slice')
    
    def getCoordinates(self, items:tuple) -> tuple:
        
        x = self.transformSizeToFitSubplots(items[0],self.nrows)
        y = self.transformSizeToFitSubplots(items[1],self.ncols)
        
        return (x,y)
    
    # def toConcreteSlices(self, items) -> tuple:
    #     l=[]
    #     for i, max in zip(items, [self.nrows, self.ncols]):
    #         if type(i) == int: ret = self.concretize(slice(i,i+1), max)
    #         elif type(i) == slice: ret = self.concretize(i, max)
    #         else: raise TypeError('elements of "items" tuple must be int or slice')
    #         l.append(ret)
    #     return tuple(l)
    
    class shareXax():
        def __init__(self):
            # super().__init__()
            pass
        def __getitem__(super, items):
            super.cols = items
    
    def shareXaxisold(self, *axeses): # join
        """
        ### Example
        myfunc.shareXaxis(myfunc[:,:])
        """
        
        # left = []
        # right = []
        
        # for axes in axeses:
        #     if type(axes) == Axes:
        #         left.append(axes.get_xlim())
        #     elif type(axes) == tuple or type(axes) == list:
        #         left.append(self.gs.__getitem__(axes).get_position(self.fig))
        #     else: raise TypeError('there must be Axes, tuple or lsit in joinXaxis')
        
        # Determine values
        # maximum = max(right)
        # minimum = min(left)
        
        # print(axeses)
        # print(axeses[0])
        if(type(axeses) == tuple) and (len(axeses) == 1): axeses = axeses[0]
        axeses[0].get_shared_x_axes()#.join(axeses)
        
    def transpose(self):
        
        # Initialize
        nrows = self.ncols # change
        ncols = self.nrows # change
        
        # Prepare gridspec & table & fig to the Axis rearrange
        gs = gridspec.GridSpec(nrows, ncols)
        if self.nrows==0 or self.ncols==0: # init if it was 0 so far
            gs.tight_layout(self.fig)
        table = np.zeros((nrows, ncols),dtype=np.int64)
        self.fig.set_size_inches(6.4*ncols, 4.8*nrows)
        
        # Axes rearrange
        for i, ax, geo in zip(range(len(self.geometry)), self.fig.axes, self.geometry):
            pos = gs[(geo[1],geo[0])].get_position(self.fig) # Bbox # change
            ax.set_position(pos)
            ax.set_subplotspec(gs[(geo[1],geo[0])]) # change
            table = self.table_update(table, (geo[1],geo[0]), i+1) # change
        
        self.nrows, self.ncols = nrows, ncols
        self.gs = gs
        self.table = table
        self.row_names, self.col_names = [___ for ___ in self.col_names], [___ for ___ in self.row_names] # change
    
    def ax(self,row,column,forced=False,log=True) -> Axes:
        """
        ### Parameters
        forced_plot (bool): whether to create axes, if another axes overlaps the location
        log (bool): whether to print warning message, if another axes overlaps the location
        """
        
        items = (row,column)
        
        items = self.getCoordinates(items) # Transform items to concrete bounds
        
        if items in self.geometry:
            return self.get_axes(items)
        else:
            return self.addaxes(items, log=log)

    def rownames(self, positions:list, names:list):
        
        # Check
        if type(positions) != list and type(positions) != range: positions = [positions]
        if type(names) != list: names = [names]
        if len(positions) != len(names): raise ValueError('length of position and length of names must be the same')
        
        for position, name in zip(positions, names):
            if name != None:
                # ax = self.__getitem__((position,0))
                self.row_names[position] = name
    
    def colnames(self, positions:list, names:list):
        
        # Check
        if type(positions) != list and type(positions) != range: positions = [positions]
        if type(names) != list: names = [names]
        if len(positions) != len(names): raise ValueError('length of position and length of names must be the same')
        
        for position, name in zip(positions, names):
            if name != None:
                # ax = self.__getitem__((0,position))
                self.col_names[position] = name

    def textsize(self, type):
        d = {'suptitle':2,
             'supaxis':1,
             'title':1,
             'axis':.5
             }
        return d[type] * sqrt(self.nrows**2 + self.ncols**2) * 3
    
    class colorbar():
        def __init__(self):
            pass
        def __getitem__(self, items):
            area = self.area(items)
            return self.fig.colorbar(self.fig.axes[0].collections[0], ax=items, orientation='vertical', label=None)
    
    def add_colorbar(self, imshow: Union[AxesImage, ContourSet], orientation: str = 'vertical', label: str = None):
        """
        Add colorbar axes to the side of the figure or below the figure.

        Parameters:
        - imshow (AxesImage or ContourSet): The image or contour set to which the colorbar corresponds.
        - orientation (str): The orientation and position of the colorbar. Can be 'vertical' or 'horizontal'. Default is 'vertical'.
        - label (str): The label of the colorbar. Default is None.
        """
        
        # Initialize subplot. It must be AxesImage or ContourSet.
        if (type(imshow) is not AxesImage) and \
            (type(imshow) is not ContourSet):
            raise ValueError('imshow must be AxesImage or ContourSet')
        
        # Add colorbar axes
        # if orientation == 'vertical':
        #     self.expand(self.nrows, self.ncols+1)
        #     location = (slice(None),-1)
        #     print('location', location)
        #     print('nrows, ncols', self.nrows, self.ncols)
        #     location = self.getCoordinates(location) # Transform items to concrete bounds
        # elif orientation == 'horizontal':
        #     self.expand(self.nrows+1, self.ncols)
        #     location = (-1,slice(None))
        #     location = self.getCoordinates(location)

        if orientation == 'vertical':
            # Get the horizontal position of the colorbar
            col = self.ncols
            h_pos = 0.9 + (col)*0.01
            dx = 0.95 - h_pos
            # Add colorbar axes
            cbar_ax = self.fig.add_axes([h_pos, 0.15, dx, 0.7])
            # Add colorbar
            cbar = self.fig.colorbar(imshow, cax=cbar_ax, orientation=orientation, label=label)
            
        elif orientation == 'horizontal':
            raise NotImplementedError('Horizontal colorbar is not implemented yet, bcs of the xlabel would overlap with the colorbar.')
            # Get the vertical position of the colorbar
            row = self.nrows
            dy = row/10
            # Add colorbar axes
            cbar_ax = self.fig.add_axes([0.15, 0.05, 0.7, dy])
            # Add colorbar
            cbar = self.fig.colorbar(imshow, cax=cbar_ax, orientation=orientation, label=label)
        
        return cbar
        
"""
EZ MŰKÖDIK:
import numpy as np
import matplotlib.pyplot as plt

t= np.arange(100)/100.
x = np.sin(2*np.pi*10*t)
y = np.cos(2*np.pi*10*t)

fig=plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.plot(t+1,x)
ax2.plot(t,y)

ax2.get_shared_x_axes().join(ax1, ax2)
# ax1.set_xticklabels([])
# ax2.autoscale() ## call autoscale if needed

plt.show()
forrás: https://stackoverflow.com/questions/42973223/how-to-share-x-axes-of-two-subplots-after-they-have-been-created
"""


# # Create 2x2 figure and two Axes at [0,:] and [1,0]
# myfig = megaplot(2,2,constrained_layout=True,title='title',xlabel='axis')
# ax = myfig[0,:]
# ax.plot([2,1,3])
# ax = myfig[1,0]
# ax.plot([2,1,3])

# # myfig.show()

# # Expand figure size to 3x3 (rows=3, cols+=1)
# myfig.expand(3,myfig.ncols+1)

# # Create a new Axes in the new figure named 'new Axes'
# ax = myfig[:,2]
# ax.plot([3,1,2])
# ax.set_title('new Axes')

# # Print myfig.table in the terminal
# print(myfig.table)
# # myfig.show()

# # Modify Axes at [1,0]
# ax = myfig[1,0]
# ax.plot([3,3,2])
# ax.set_title('modified Axes')

# # Create a new Axes ([0,1]) in an alternative mode on top of an existing (our first Axes)
# ax = myfig.ax(0,1)
# ax.plot([10,2])
# ax.set_title('displaced Axes')
# # ax = myfig[2,0]
# # ax.plot([3,4,5],[3,3,4])

# # myfig.shareXaxis(myfig[1:,0])
# # myfig.shareXaxis[10]
# # print(myfig.cols)

# # myfig[1,1] = plt.scatter([.67,1.33],[1.5,1.5]) # ezzel nem lehet megoldani

# print(myfig[0,:])

# # print(Axes.get_shared_x_axes().join(ax1,ax2))

# # Create titles for the rows and the columns
# myfig.rownames([0,1,2],['a','b','new row'])
# myfig.colnames([0,1,2],['a','b','new column'])

# # Transpose the whole figure (change columns and rows)
# myfig.transpose()

# # Show our figure
# myfig.show()

# # Save our figure
# # myfig.savefig('test',type='pdf',path='test')


"""
print()
# https://matplotlib.org/stable/api/gridspec_api.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html#matplotlib.gridspec.GridSpec
# https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.SubplotSpec.html#matplotlib.gridspec.SubplotSpec
# subfigs = fig.subfigures(1, 2, wspace=0.07) # matplotlib.figure.SubFigure # A subfigure has the same artist methods as a figure, and is logically the same as a figure, but cannot print itself. see: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html
# matplotlib.pyplot.subplot_mosaic
# allaxes = fig.get_axes()
fig = plt.figure()
fig.add_subplot() # Create a background empty plot
gs = fig.add_gridspec(2,2) # return matplotlib.gridspec.GridSpec
gs.get_geometry() # return tuple with rows, cols
SS = gs[0] # return matplotlib.gridspec.SubplotSpec
SS = gs[0,0:2] # return matplotlib.gridspec.SubplotSpec #exclusive
ax = fig.add_subplot(SS)
ax.get_subplotspec() # returns matplotlib.gridspec.SubplotSpec of the subplot
ax.set_subplotspec(gs[1,0]) # set the subplot to a new position
SS.get_position(fig) # return matplotlib.transforms.Bbox
SS.is_last_row() # count from left upper corner
innergs = SS.subgridspec(3,3) # matplotlib.gridspec.GridSpecFromSubplotSpec
haha = fig.add_subplot(innergs[1,1]) # matplotlib.axes._subplots.AxesSubplot

# Subplot of subplot:
# innerSS = innergs[:,2] # matplotlib.gridspec.GridSpecFromSubplotSpec
# innestgs = innerSS.subgridspec(2,1) # matplotlib.gridspec.GridSpecFromSubplotSpec
# naesez = fig.add_subplot(innestgs[0]) # matplotlib.axes._subplots.AxesSubplot

gridspec = matplotlib.gridspec # this is just a library or directory
plt.gca() # get the currently active axes

plt.show()
"""


# Protips:
# gs_kw = dict(width_ratios=[1.4, 1], height_ratios=[1, 2]) # at 2x2 fig (https://www.adamsmith.haus/python/answers/how-to-set-subplot-size-in-matplotlib-in-python)
# gridspec_kw=gs_kw
# subfigs[1].suptitle('subfigs[1]')
# subfigs[1].supylabel('ylabel for subfigs[1]')
# ax.lines[0].get_ydata
# https://stackoverflow.com/questions/14770735/how-do-i-change-the-figure-size-with-subplots



# vmi = subplot(2,2)
# (0,slice(None,None,None)):
# hm = vmi.toConcreteSlices((0,slice(None,None,None)))
# print(hm)
# for hely in vmi.area(hm):
#     print(hely)

