import sys
from optparse import OptionParser
import pylab
import matplotlib
import numpy as np
import preprocessing
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from scipy.stats import *
 
def importData(filename):
    matrix = []
    row_header = []
    column_header = []
    first_row = True
 
    if '/' in filename:
        dataset_name = filename.split('/')[-1]
    else:
        dataset_name = filename
 
    for line in open(filename,'r'):
        t = line.rstrip().split('\t')
        if first_row:
            column_header = t[1:]
            first_row = False
        else:
            if ' ' not in t and '' not in t:
                s = t[1:]
                s = [ float(x) for x in s ]
                if (abs(max(s)-min(s))) > 0:
                    matrix.append(s)
                    row_header.append(t[0])
 
    try:
        print '\n%d rows and %d columns imported for %s...\n' % (len(matrix), len(column_header), dataset_name)
    except Exception:
        print 'No data in input file.'; force_error
 
    matrix = preprocessing.relative_abundance(matrix,axis=0)
 
    return np.array(matrix), row_header, column_header
 
 
def executePCA(matrix, row_header, column_header, drawBiplot, threshold, not_scale):
 
    tdata = matrix.T
 
    if not_scale:
        sdata = tdata
    else:
        # standardizing data
        sdata = scale(tdata)
 
    # executing PCA
    pca = PCA()
    pca.fit(sdata)
 
    print 'explained variance ratios ...'
    print [ 100.0*v for v in pca.explained_variance_ratio_ ]
    print '\n'
    pc1explain = 100.0 * pca.explained_variance_ratio_[0]
    pc2explain = 100.0 * pca.explained_variance_ratio_[1]
 
    pcCoord = pca.fit_transform(sdata)
 
    # draw barchart of factor loadings
    pc1Cors = []
    pc2Cors = []
    for i, varName in enumerate(row_header):
        values = matrix[i]
        vsPC1 = pearsonr(values, pcCoord[:,0])[0]
        vsPC2 = pearsonr(values, pcCoord[:,1])[0]
        pc1Cors.append(vsPC1)
        pc2Cors.append(vsPC2)
 
    ofp = open('factorLoadings_pc1.list','w')
    for cor, var in zip(pc1Cors, row_header):
        ofp.write(str(cor)+'\t'+var+'\n')
    ofp.close()
 
    ofp = open('factorLoadings_pc2.list','w')
    for cor, var in zip(pc2Cors, row_header):
        ofp.write(str(cor)+'\t'+var+'\n')
    ofp.close()
 
    bpcolors = ['b'] * len(row_header)
 
    pc1list = [pc1Cors, bpcolors, row_header]
    pc1list = zip(*pc1list)
    pc1list = [ x for x in pc1list if np.abs(x[0]) >= threshold ]
    pc1list.sort(reverse=True)
    fig = pylab.figure(figsize=(16,24))
    fig.add_subplot(411)
    index = np.arange(len(pc1list))
    width = 0.5
    pylab.bar(index, [x[0] for x in pc1list], width, color=[x[1] for x in pc1list])
    pylab.ylim(-1.0,1.0)
    pylab.title('factor loadings for PC1')
    pylab.xticks(index+width/2., [x[2] for x in pc1list], rotation=-90, fontsize=12)
    pylab.grid(True)
 
    pc2list = [pc2Cors, bpcolors, row_header]
    pc2list = zip(*pc2list)
    pc2list = [ x for x in pc2list if np.abs(x[0]) >= threshold ]
    pc2list.sort(reverse=True)
    fig.add_subplot(413)
    index = np.arange(len(pc2list))
    width = 0.5
    pylab.bar(index, [x[0] for x in pc2list], width, color=[x[1] for x in pc2list])
    pylab.ylim(-1.0,1.0)
    pylab.title('factor loadings for PC2')
    pylab.xticks(index+width/2., [x[2] for x in pc2list], rotation=-90, fontsize=12)
    pylab.grid(True)
 
    pylab.savefig('factorLoadings_pc12.png')
    pylab.clf()
 
 
    # draw figure PC1 vs PC2
    pylab.figure(figsize=(12,12))
    pc12 = []
    pc12.append(pcCoord[:,0])
    pc12.append(pcCoord[:,1])
    if np.abs(np.min(pc12)) > np.abs(np.max(pc12)):
        maxCoord = np.abs(np.min(pc12)) * 1.1
    else:
        maxCoord = np.abs(np.max(pc12)) * 1.1
    pylab.xlim(-1 * maxCoord, maxCoord)
    pylab.ylim(-1 * maxCoord, maxCoord)
 
    ax = pylab.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
 
    if drawBiplot:
        circle = pylab.Circle((0,0), radius=maxCoord, fc='none', linestyle='dashed', color='gray')
        pylab.gca().add_patch(circle)
        for i,varName in enumerate(row_header):
            xcoord = pc1Cors[i] * maxCoord
            ycoord = pc2Cors[i] * maxCoord
            length = np.sqrt(pc1Cors[i]**2 + pc2Cors[i]**2)
            if length <= threshold:
                continue
            arrowsize = maxCoord / 100.0
            ax.arrow(0.0, 0.0, xcoord, ycoord, head_width=arrowsize, head_length=2*arrowsize, fc=bpcolors[i], ec=bpcolors[i], alpha=0.5)
            pylab.annotate(varName, xy=(xcoord,ycoord), xytext=(5,5), textcoords='offset points', color=bpcolors[i], fontsize=12)
 
 
    for i, sample in enumerate(column_header):
        pylab.annotate(sample, xy=(pcCoord[i,0], pcCoord[i,1]), xytext=(5,5), textcoords='offset points', color='k', fontsize=16)
 
    pylab.scatter(pcCoord[:,0], pcCoord[:,1], c='k', s=50)
 
    x_label = 'PC1 (%.2f%%)'%(pc1explain)
    y_label = 'PC2 (%.2f%%)'%(pc2explain)
    label_position = maxCoord
    pylab.annotate(x_label, xy=(0.0, -1*label_position), xytext=(0.0,-40.0), textcoords='offset points', ha='center', color='k', fontsize=18)
    pylab.annotate(y_label, xy=(-1*label_position, 0.0), xytext=(-40.0,50.0), textcoords='offset points', ha='center', color='k', fontsize=18, rotation=90)
 
    pylab.savefig('x_pc1_y_pc2.png')
    pylab.clf()
 
 
if __name__ == '__main__':
 
    usage = "usage: python %prog [options]"
 
    parser = OptionParser(usage)
 
    parser.add_option( "-f", "--file", action="store", dest="data_file", help="matrix data file. rows are variables, columns are samples.")
    parser.add_option( "-b", "--biplot", action="store_true", dest="biplot", default=False, help="output biplot (PC with factor loadings).")
    parser.add_option( "-t", "--threshold_of_length", action="store", type="float", dest="threshold", default=0.0, help="length threshold when drawing biplot arrows.")
    parser.add_option( "-n", "--not_scale_data", action="store_true", dest="not_scale", default=False, help="NOT standardize data matrix. e.g. when using abundance matrix.")
 
    options, args = parser.parse_args()
 
    if options.data_file == None:
        print "ERROR: requires options"
        parser.print_help()
        sys.exit()
 
    datafile = options.data_file
    drawBiplot = options.biplot
    threshold = options.threshold
    not_scale = options.not_scale
 
    matrix, row_header, column_header = importData(datafile)
 
    executePCA(matrix, row_header, column_header, drawBiplot, threshold, not_scale)
 
    print 'done.'
