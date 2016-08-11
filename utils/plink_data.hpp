struct packedgeno_t{
  char geno[4];
  packedgeno_t(){
    for(int i=0;i<4;++i) geno[i] = static_cast<char>(0);
  }
};


class plink_data_t{
public:
  static const int * plink_geno_mapping;
  static const int PLINK_BLOCK_WIDTH=512;
  static const int ROWS = 0;
  static const int COLS = 1;
  plink_data_t(int rows,int cols);
  plink_data_t(int rowmasklen,bool * rowmask, int colmasklen,bool * colmask);
  ~plink_data_t();
  void load_data(const char * filename);
  float get_geno(int row,int col);
  int get_raw_geno(int row,int col);
  void compute_mean_precision(int axis, const char * outfile);
  void compute_allele_freq(int axis, const char * outfile);
  void set_mean_precision(int axis, float * mean_arr,float * precision_arr);
  void copy(const char * filename);
  // you can either transpose into a file or into a new PLINK data object
  void transpose(const char * filename);
  void output(const char * filename);
  // in this case we'll take care of the memory allocation
  void transpose(plink_data_t * & plink_data);
  packedgeno_t * get_packed_geno(int & packedstride);
private:
  bool snp_major_mode;
  void init();
  int get_plink_geno(int row,int col);
  packedgeno_t * packedgeno_matrix;
  int packedstride,packedgenolen;
  int totalrows,totalcols;
  int veclen;
  bool deletemask;
  bool * row_mask;
  int row_mask_len,col_mask_len;
  bool * col_mask;
  float * mean_arr;
  float * precision_arr;
  int mean_precision_axis;
  int mean_precision_len;
};
