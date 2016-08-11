#include<string.h>
#include<assert.h>
#include<math.h>
#include<iostream>
#include<sstream>
#include<fstream>
#include<cstdlib>
using namespace std;
#include"plink_data.hpp"

static const int mapping_data[]={0,9,1,2};
const int * plink_data_t::plink_geno_mapping = mapping_data;


plink_data_t::plink_data_t(int totalrows,int totalcols){
  row_mask = new bool[totalrows];
  for(int i=0;i<totalrows;++i) row_mask[i] = true;
  col_mask = new bool[totalcols];
  for(int i=0;i<totalcols;++i) col_mask[i] = true;
  deletemask = true;
  this->totalrows = this->row_mask_len = totalrows ;
  this->totalcols = this->col_mask_len = totalcols;
  init();
}

plink_data_t::plink_data_t(int rowmasklen,bool * rowmask, int colmasklen,bool * colmask){
  this->row_mask_len = rowmasklen;
  this->row_mask = rowmask;
  this->col_mask_len = colmasklen;
  this->col_mask = colmask;
  deletemask = false;
  this->totalrows = 0;
  for(int i=0;i<rowmasklen;++i) this->totalrows+=rowmask[i];
  this->totalcols = 0;
  for(int i=0;i<colmasklen;++i) this->totalcols+=colmask[i];
  cerr<<"PLINK data initializing with row mask len "<<row_mask_len<<" and col mask len "<<col_mask_len<<" and rows "<<totalrows<<" and cols "<<totalcols<<endl;
  init();
}

void plink_data_t::init(){
  mean_arr = NULL;
  precision_arr = NULL;
  mean_precision_axis = -1;
}

void plink_data_t::set_mean_precision(int axis, float * mean_arr,float  * precision_arr){
  this->mean_precision_axis = axis;
  this->mean_arr = mean_arr;
  this->precision_arr = precision_arr;
  
}

void plink_data_t::load_data(const char * filename){
  ifstream ifs;
  ifs.open(filename,ios::in|ios::binary);
  if (!ifs.is_open()){
    cerr<<"Could not open "<<filename<<". Exiting.\n";
    exit(1);
  }
  // buffer for reading in byte data
  bool remainder = (totalcols%4!=0)?true:false;
  int veclen = totalcols/4+remainder;
  char data[veclen];
  //cerr<<"Total rows: "<<totalrows<<" total cols: "<<totalcols<<" veclen: "<<veclen<<endl;
  // four byte data type
  packedstride = (totalcols/PLINK_BLOCK_WIDTH+(totalcols%PLINK_BLOCK_WIDTH>0))*PLINK_BLOCK_WIDTH/16;
  packedgenolen = totalrows*packedstride;
  packedgeno_matrix = new packedgeno_t[packedgenolen];
  //cerr<<"Packed stride is "<<packedstride<<" and matrix len: "<<packedgenolen<<endl;
  // ignore first 3 octets
  ifs.read(data,3);
  this->snp_major_mode = (bool)data[2];
  int packed_row_index = 0;
  for(int row=0;row<row_mask_len;++row){
    if(row_mask[row]){
      for(int j=0;j<veclen;++j) data[j] = 0;
      if(col_mask_len==totalcols){
        ifs.seekg(3+row*veclen);
        ifs.read(data,veclen); 
      }else{
//cerr<<"Reading in row "<<row<<endl;
        int totalshifts = 0;
        int dataindex = 0;
        for(int j=0;j<col_mask_len;++j){
          if(col_mask[j]){
//cerr<<"Reading in col "<<j<<endl;
            int byteindex = (j/4);
            //char byte = tempdata[byteindex];
            ifs.seekg(3+row*veclen+byteindex);
            char byte;
            ifs.read(&byte,1); 
            //if(row>8) cerr<<"Byte index: "<<byteindex<<" byte: "<<byte<<endl;
            int shifts = j%4;
            //cerr<<"Shifts: "<<shifts<<endl;
            int plinkgeno = ((int)byte) >> (2*shifts) & 3;
            //cerr<<"Fetching row "<<i<<" col "<<j<<" Plink geno before "<<plinkgeno<<" byte: "<<byte<<endl;
            //cerr<<"Totalshifts: "<<totalshifts<<endl;
            plinkgeno = plinkgeno << (2*totalshifts);
            //cerr<<"Plink geno shifted "<<plinkgeno<<endl;
            data[dataindex] = data[dataindex] | plinkgeno;
            //cerr<<"Data at "<<dataindex<<" is "<<(int)(unsigned char)data[dataindex]<<endl;
            if(totalshifts==3){
              totalshifts = 0;
              ++dataindex;
            }else{
              ++totalshifts;
            }
          }
        }
      }
      int col=0;
      int p=0;
      for(int chunk=0;chunk<packedstride;++chunk){
        for(int subchunk=0;subchunk<4;++subchunk){
          if (col<totalcols){
            //cerr<<"Loading into chunk "<<chunk<<" subchunk "<<subchunk<<" At p "<<p<<endl;
            packedgeno_matrix[packed_row_index*packedstride+chunk].geno[subchunk] = data[p++];
            //cerr<<"Loading into chunk "<<chunk<<" subchunk "<<subchunk<<" At p "<<p<<": "<<packedgeno_matrix[packed_row_index*packedstride+chunk].geno[subchunk]<<endl;
            col+=4;
          }else{
            //cerr<<"Finished At col "<<col<<endl;
          }
        }
      }
      for(int j=0;j<totalcols;++j){
        //cerr<<get_raw_geno(packed_row_index,j);
      }
      ++packed_row_index;
     
      //cerr<<"Row "<<row<<" parsed\n";
    //cerr<<endl;
    }else{
      //cerr<<"Row "<<row<<" skipped\n";
    }
    
  }
  ifs.close();
  //cerr<<"PLINK data loaded.\n";
}

plink_data_t::~plink_data_t(){
  if (deletemask){
    delete[] row_mask;
    delete[] col_mask;
  }
  delete [] packedgeno_matrix;
}

void plink_data_t::transpose(plink_data_t * & transposed){
  transposed = new plink_data_t(totalcols,totalrows);
  int veclen = totalrows/4+(totalrows%4!=0);
  char data[veclen];
  transposed->packedstride = (totalrows/PLINK_BLOCK_WIDTH+(totalrows%PLINK_BLOCK_WIDTH>0))
  * PLINK_BLOCK_WIDTH/16;
  transposed->packedgenolen = totalcols * transposed->packedstride;
  //cerr<<"Packed geno len of transpose is "<<transposed->packedgenolen<<endl;

  transposed->packedgeno_matrix = new packedgeno_t[transposed->packedgenolen];
  int packed_row_index = 0; 
  for(int j=0;j<totalcols;++j){
    int totalshifts = 0;
    int dataindex = 0;
    for(int i=0;i<veclen;++i) data[i] = (char)0;
    for(int i=0;i<totalrows;++i){
      int plinkgeno = get_plink_geno(i,j);
      plinkgeno = plinkgeno << (2*totalshifts);
      data[dataindex] = data[dataindex] | plinkgeno;
      if(totalshifts==3){
        totalshifts = 0;
        ++dataindex;
      }else{
        ++totalshifts;
      }
    }
    //cerr<<"Data index at "<<dataindex<<endl;
    int col=0;
    int p=0;
    for(int chunk=0;chunk<transposed->packedstride;++chunk){
      for(int subchunk=0;subchunk<4;++subchunk){
        if (col<totalrows){
          //cerr<<"Loading into chunk "<<chunk<<" subchunk "<<subchunk<<" at p "<<p<<": "<<endl;
          //cerr<<" "<<packed_row_index*transposed->packedstride+chunk;
          transposed->packedgeno_matrix[packed_row_index*transposed->packedstride+chunk].geno[subchunk] = data[p++];
          //char c = data[p++];
          col+=4;
        }else{
          //cerr<<"Finished At col "<<col<<endl;
        }
      }
    }
    //cerr<<endl;
    ++packed_row_index;
    //cerr<<"Col "<<j<<" done\n";
  }
  transposed->mean_arr = this->mean_arr;
  transposed->precision_arr = this->precision_arr;
  if(this->mean_precision_axis==ROWS){
    transposed->mean_precision_axis = COLS;
  }else if(this->mean_precision_axis==COLS){
    transposed->mean_precision_axis = ROWS;
  }
}

packedgeno_t * plink_data_t::get_packed_geno(int & packedstride){
  packedstride = this->packedstride;
  return packedgeno_matrix;
}

void plink_data_t::output(const char * filename){
  ofstream ofs(filename);
  char plink_header[3];
  plink_header[0] = 108;
  plink_header[1] = 72;
  plink_header[2] = snp_major_mode;
  ofs.write(plink_header,3);
  int veclen = totalcols/4+(totalcols%4!=0);
  char data[veclen];
  //cerr<<"Vec len is "<<veclen<<endl;
  for(int j=0;j<totalrows;++j){
    int totalshifts = 0;
    int dataindex = 0;
    for(int i=0;i<veclen;++i) data[i] = (char)0;
    for(int i=0;i<totalcols;++i){
      int plinkgeno = get_plink_geno(j,i);
      //cerr<<" "<<j<<","<<i<<":"<<plinkgeno;
      plinkgeno = plinkgeno << (2*totalshifts);
      data[dataindex] = data[dataindex] | plinkgeno;
      if(totalshifts==3){
        totalshifts = 0;
        ++dataindex;
      }else{
        ++totalshifts;
      }
    }
    ofs.write(data,veclen);
  }
  ofs.close();
}

void plink_data_t::copy(const char * filename){
  cerr<<"Doing copy !\n";
  ofstream ofs(filename);
  char plink_header[3];
  plink_header[0] = 108;
  plink_header[1] = 27;
  //plink_header[1] = 72;
  plink_header[2] = snp_major_mode;
  ofs.write(plink_header,3);
  int veclen = totalcols/4+(totalcols%4!=0);
  char data[veclen];
  //cerr<<"Vec len is "<<veclen<<endl;
  for(int j=0;j<totalrows;++j){
    int totalshifts = 0;
    int dataindex = 0;
    for(int i=0;i<veclen;++i) data[i] = (char)0;
    for(int i=0;i<totalcols;++i){
      int plinkgeno = get_plink_geno(j,i);
//cerr<<"Plink geno at "<<i<<","<<j<<" before :"<<plinkgeno<<endl;
      plinkgeno = plinkgeno << (2*totalshifts);
//cerr<<" after "<<totalshifts<<" shifts: "<<(int)plinkgeno<<" data "<<(int)((unsigned char)data[dataindex])<<"\n"; 
      data[dataindex] = data[dataindex] | plinkgeno;
//cerr<<" data index at "<<dataindex<<": "<<(int)((unsigned char)data[dataindex])<<endl;
      if(totalshifts==3){
        totalshifts = 0;
        ++dataindex;
      }else{
        ++totalshifts;
      }
    }
    ofs.write(data,veclen);
    //cerr<<"Col "<<j<<" done\n";
  }
  ofs.close();
}

void plink_data_t::transpose(const char * filename){
  ofstream ofs(filename);
  char plink_header[3];
  plink_header[0] = 108;
  plink_header[1] = 72;
  plink_header[2] = !snp_major_mode;
  ofs.write(plink_header,3);
  int veclen = totalrows/4+(totalrows%4!=0);
  char data[veclen];
  //cerr<<"Vec len is "<<veclen<<endl;
  for(int j=0;j<totalcols;++j){
    int totalshifts = 0;
    int dataindex = 0;
    for(int i=0;i<veclen;++i) data[i] = (char)0;
    for(int i=0;i<totalrows;++i){
      int plinkgeno = get_plink_geno(i,j);
//cerr<<"Plink geno at "<<i<<","<<j<<" before :"<<plinkgeno<<endl;
      plinkgeno = plinkgeno << (2*totalshifts);
//cerr<<" after "<<totalshifts<<" shifts: "<<(int)plinkgeno<<" data "<<(int)((unsigned char)data[dataindex])<<"\n"; 
      data[dataindex] = data[dataindex] | plinkgeno;
//cerr<<" data index at "<<dataindex<<": "<<(int)((unsigned char)data[dataindex])<<endl;
      if(totalshifts==3){
        totalshifts = 0;
        ++dataindex;
      }else{
        ++totalshifts;
      }
    }
    ofs.write(data,veclen);
    //cerr<<"Col "<<j<<" done\n";
  }
  ofs.close();
}
  
void plink_data_t::compute_allele_freq(int axis, const char * outfile){
  int x,y;
  if(axis==ROWS){
    x = totalrows;
    y = totalcols;
  }else if(axis==COLS){
    x = totalcols;
    y = totalrows;
  }else{
    cerr<<"Unknown axis of "<<axis<<endl;
    exit(1);
  }
  ofstream ofs(outfile);
  for(int i=0;i<x;++i){
    int n=0;
    float freq = 0;
    int g;
    for(int j=0;j<y;++j){
      g = axis==ROWS?get_raw_geno(i,j):get_raw_geno(j,i);
      if(g!=9){ 
        freq+=g;
        ++n;
      }
    }
    freq/=2*n;
    ofs<<i<<"\t"<<freq<<endl;
  }
  ofs.close();
}

void plink_data_t::compute_mean_precision(int axis, const char * outfile){
  int x,y;
  this->mean_precision_axis = axis;
  if(axis==ROWS){
    x = totalrows;
    y = totalcols;
  }else if(axis==COLS){
    x = totalcols;
    y = totalrows;
  }else{
    cerr<<"Unknown axis of "<<axis<<endl;
    exit(1);
  }
  mean_arr = new float[x];
  precision_arr = new float[x];
  ofstream ofs(outfile);
  for(int i=0;i<x;++i){
    mean_arr[i] = 0;
    precision_arr[i] = 0;
    int n=0;
    int g;
    for(int j=0;j<y;++j){
      g = axis==ROWS?get_raw_geno(i,j):get_raw_geno(j,i);
      if(g!=9){ 
        mean_arr[i]+=g;
        ++n;
      }
    }
    mean_arr[i]/=n;
    for(int j=0;j<y;++j){
      g = axis==ROWS?get_raw_geno(i,j):get_raw_geno(j,i);
      if(g!=9){ 
        float dev = g-mean_arr[i];
        precision_arr[i]+=dev*dev;
      }
    }
    precision_arr[i] = precision_arr[i]==0?0:sqrt((n-1)/precision_arr[i]);
    //cerr<<"Row "<<i<<" standardized.\n";
    ofs<<mean_arr[i]<<"\t"<<precision_arr[i]<<endl;
  }
  ofs.close();
}

int plink_data_t::get_plink_geno(int row,int col){
  bool debug = false;
  //if (col==totalcols-1) debug = true;
  if (debug) cerr<<"Fetching row "<<row<<" col "<<col<<endl;
  int packedindex = col/16;
  if (debug)cerr<<"Packed geno at "<<row*packedstride+packedindex<<endl;
  packedgeno_t packedgeno = packedgeno_matrix[row*packedstride+packedindex];
  int byteindex = (col - packedindex*16)/4;
  if (debug)cerr<<"Byte at "<<byteindex<<endl;
  char byte = packedgeno.geno[byteindex];
  int shifts = col%4;
  if (debug)cerr<<"Shifts: "<<shifts<<endl;
  int geno = ((int)byte) >> (2 * shifts) & 3;
  assert(geno>=0 && geno<=3);
  if (debug)cerr<<"Geno: "<<geno<<endl;
  
  return geno;
}

int plink_data_t::get_raw_geno(int row,int col){
  return plink_geno_mapping[get_plink_geno(row,col)];
}

float plink_data_t::get_geno(int row,int col){
  int rawgeno = get_raw_geno(row,col);
  //cerr<<"Raw geno "<<rawgeno<<endl;
  float standardized = 0;
  if(mean_arr!=NULL && precision_arr!=NULL && rawgeno!=9){
    if (mean_precision_axis==ROWS){
      //cerr<<"A Fetching from mean arr "<<row<<endl;
      //cerr<<";"<<rawgeno<<","<<mean_arr[row]<<","<<precision_arr[row];
      standardized = (rawgeno-mean_arr[row])*precision_arr[row];
      //cerr<<"At row "<<row<<", Returning value: raw: "<<rawgeno<<" mean "<<mean_arr[row]<<" prec "<<precision_arr[row]<<endl;
      if(isnan(standardized) || isinf(standardized)){
        //cerr<<"At row "<<row<<", Returning invalid value: raw: "<<rawgeno<<" mean "<<mean_arr[row]<<" prec "<<precision_arr[row]<<endl;
      }
    }else if (mean_precision_axis==COLS){
      //cerr<<"B Fetching from mean arr "<<col<<endl;
      standardized = (rawgeno-mean_arr[col])*precision_arr[col];
    }else{
      cerr<<"Skipping as axis not defined\n";
    }
  }
  return standardized;
}

void read_mask(const char * file, int len,bool * mask){
  ifstream ifs(file);
  if (!ifs.is_open()){
    cerr<<"Cannot open "<<file<<". Exiting\n";
    exit(1);
  }
  for(int i=0;i<len;++i){
    string line;
    getline(ifs,line);
    istringstream iss(line);
    iss>>mask[i];
  }
  ifs.close();
}

int main(int argc,char * argv[]){
  if(argc<5){
    cerr<<"Usage: <filename> <rows> <cols> <options>\nOptions:\n";
    cerr<<"--transpose <transposed file>\n";
    cerr<<"--copy <new file>\n";
    cerr<<"--summary [rows|cols] <maf file> <mean/precision file>\n";
    cerr<<"--mask <rowmask_file> <colmask_file>\n";
    exit(0);
  }
  int arg = 0;
  const char * filename = argv[++arg];
  int totalrows = atoi(argv[++arg]);
  int totalcols = atoi(argv[++arg]);
  bool * rowmask = new bool[totalrows];
  bool * colmask = new bool[totalcols];
  for(int i=0;i<totalrows;++i) rowmask[i] = false;
  for(int i=0;i<totalcols;++i) colmask[i] = false;
  plink_data_t * plink_data = NULL;
  bool enable_mask = false;
  bool enable_transpose = false;
  bool enable_copy = false;
  const char * transpose_file= NULL;
  const char * copy_file= NULL;
  bool enable_summary = false;
  int mean_precision_axis = 0;
  const char * maf_file= NULL;
  const char * mean_precision_file= NULL;
  ++arg;
  while(arg<argc){
    if(strcmp(argv[arg],"--transpose")==0){
      enable_transpose = true;
      transpose_file=argv[++arg];
      cerr<<"Invoking transpose with outfile "<<transpose_file<<"\n";
    }else if(strcmp(argv[arg],"--copy")==0){
      enable_copy = true;
      copy_file=argv[++arg];
      cerr<<"Invoking copy with outfile "<<copy_file<<"\n";
    }else if(strcmp(argv[arg],"--summary")==0){
      enable_summary = true;
      const char * axis = argv[++arg];
      if (strcmp(axis,"rows")==0){
        mean_precision_axis = plink_data_t::ROWS;
      }else if (strcmp(axis,"cols")==0){
        mean_precision_axis = plink_data_t::COLS;
      }else{
        cerr<<"Invalid axis of "<<axis<<endl;
        return 1;
      }
      maf_file=argv[++arg];
      mean_precision_file=argv[++arg];
      cerr<<"Invoking summary stats with outfiles "<<maf_file<<" and "<<mean_precision_file<<"\n";
    }else if(strcmp(argv[arg],"--mask")==0){
      enable_mask = true;
      const char * rowfile=argv[++arg];
      const char * colfile=argv[++arg];
      cerr<<"Invoking mask with files "<<rowfile<<" and "<<colfile<<"\n";
      read_mask(rowfile,totalrows,rowmask);
      read_mask(colfile,totalcols,colmask);
    }
    ++arg;
  }
  if(enable_mask){
    //for(int i=0;i<37547;++i) rowmask[i] = true;
    //for(int j=0;j<2000;++j) colmask[j] = true;
    plink_data = new plink_data_t(totalrows,rowmask,totalcols,colmask);
  }else{
    plink_data = new plink_data_t(totalrows,totalcols);
  }
  plink_data->load_data(filename);
  if(enable_transpose){
    plink_data->transpose(transpose_file);
  }
  if(enable_copy){
    plink_data->copy(copy_file);
  }
  //plink_data->output("wtccc-n2k_chr1.bed");
  if(enable_summary){
    plink_data->compute_allele_freq(mean_precision_axis,maf_file);
    plink_data->compute_mean_precision(mean_precision_axis,mean_precision_file);
  }
  //plink_data_t * transposed;
  //plink_data->transpose(transposed);
  //int j = 30;
  //cerr<<"I: "<<i<<" J: "<<j<<endl;
  //cerr<<"Get geno: ";
  for(int i=0;i<1000;++i){
    //cerr<<plink_data->get_raw_geno(j,i);
  }
  //cerr<<endl;
  //cerr<<"Transposed "<<transposed->get_raw_geno(i,j)<<endl;
  
  //cerr<<"Get geno "<<plink_data->get_geno(j,i)<<endl;
  //cerr<<"Transposed "<<transposed->get_geno(i,j)<<endl;

  delete plink_data;
  delete[]rowmask;
  delete[]colmask;
  return 0;
}
