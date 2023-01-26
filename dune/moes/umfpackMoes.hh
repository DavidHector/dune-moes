#ifndef DUNE_MOES_UMFPACKMOES_HH
#define DUNE_MOES_UMFPACKMOES_HH
// #if HAVE_SUITESPARSE_UMFPACK || defined DOXYGEN

#include <dune/moes/vectorclass/vectorclass.h>
#include <iostream>

#include <complex>
#include <type_traits>

#include <umfpack.h>
#include <dune/moes/Utils.hh>

#include <dune/common/exceptions.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/istl/bccsmatrixinitializer.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/solvertype.hh>
#include <dune/istl/solverfactory.hh>

/* NOTE: 
How to do LU-decomposition? 
    - umfpack_*_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, Info); 
    - umfpack_*_get_numeric() basically takes the Numeric object from before and gives the LU factors

*/

/* TODO: 
1. add the umfpack_*_get_numeric to the MethodChooser
2. implement an alternative solve function using the L-U factors on my data structure

*/
namespace Dune
{
    /**
   * @addtogroup MOES
   *
   * @{
   */
    /**
   * @file
   * @author David Hector (adapted from umfpack.hh by Dominic Kempf)
   * @brief Classes for using UMFPackMOES with MOES data structures and matrices.
   */

    // FORWARD DECLARATIONS
    template <class M, class T, class TM, class TD, class TA>
    class SeqOverlappingSchwarz;

    template <class T, bool tag>
    struct SeqOverlappingSchwarzAssemblerHelper;

    // wrapper class for C-Function Calls in the backend. Choose the right function namespace
    // depending on the template parameter used.
    template <typename T>
    struct UMFPackMethodChooser
    {
        static constexpr bool valid = false;
    };

    template <>
    struct UMFPackMethodChooser<double>
    {
        static constexpr bool valid = true;

        template <typename... A>
        static void defaults(A... args)
        {
            umfpack_dl_defaults(args...);
        }
        template <typename... A>
        static void free_numeric(A... args)
        {
            umfpack_dl_free_numeric(args...);
        }
        template <typename... A>
        static void free_symbolic(A... args)
        {
            umfpack_dl_free_symbolic(args...);
        }
        template <typename... A>
        static int load_numeric(A... args)
        {
            return umfpack_dl_load_numeric(args...);
        }
        template <typename... A>
        static void numeric(A... args)
        {
            umfpack_dl_numeric(args...);
        }
        template <typename... A>
        static int get_numeric(A... args)
        {
            return umfpack_dl_get_numeric(args...);
        }
        template <typename... A>
        static int get_lunz(A... args)
        {
            return umfpack_dl_get_lunz(args...);
        }
        template <typename... A>
        static void report_info(A... args)
        {
            umfpack_dl_report_info(args...);
        }
        template <typename... A>
        static void report_status(A... args)
        {
            umfpack_dl_report_status(args...);
        }
        template <typename... A>
        static int save_numeric(A... args)
        {
            return umfpack_dl_save_numeric(args...);
        }
        template <typename... A>
        static void solve(A... args)
        {
            umfpack_dl_solve(args...);
        }
        template <typename... A>
        static void symbolic(A... args)
        {
            umfpack_dl_symbolic(args...);
        }
    };

    template <>
    struct UMFPackMethodChooser<std::complex<double>>
    {
        static constexpr bool valid = true;

        template <typename... A>
        static void defaults(A... args)
        {
            umfpack_zl_defaults(args...);
        }
        template <typename... A>
        static void free_numeric(A... args)
        {
            umfpack_zl_free_numeric(args...);
        }
        template <typename... A>
        static void free_symbolic(A... args)
        {
            umfpack_zl_free_symbolic(args...);
        }
        template <typename... A>
        static int load_numeric(A... args)
        {
            return umfpack_zl_load_numeric(args...);
        }
        template <typename... A>
        static void numeric(const long int *cs, const long int *ri, const double *val, A... args)
        {
            umfpack_zl_numeric(cs, ri, val, NULL, args...);
        }
        template <typename... A>
        static int get_numeric(A... args)
        {
            return umfpack_zl_get_numeric(args...);
        }
        template <typename... A>
        static int get_lunz(A... args)
        {
            return umfpack_zl_get_lunz(args...);
        }
        template <typename... A>
        static void report_info(A... args)
        {
            umfpack_zl_report_info(args...);
        }
        template <typename... A>
        static void report_status(A... args)
        {
            umfpack_zl_report_status(args...);
        }
        template <typename... A>
        static int save_numeric(A... args)
        {
            return umfpack_zl_save_numeric(args...);
        }
        template <typename... A>
        static void solve(long int m, const long int *cs, const long int *ri, std::complex<double> *val, double *x, const double *b, A... args)
        {
            const double *cval = reinterpret_cast<const double *>(val);
            umfpack_zl_solve(m, cs, ri, cval, NULL, x, NULL, b, NULL, args...);
        }
        template <typename... A>
        static void symbolic(long int m, long int n, const long int *cs, const long int *ri, const double *val, A... args)
        {
            umfpack_zl_symbolic(m, n, cs, ri, val, NULL, args...);
        }
    };

    namespace Impl
    {
        template <class M>
        struct UMFPackVectorChooser
        {
        };

        template <typename T, typename A, int n, int m>
        struct UMFPackVectorChooser<BCRSMatrix<FieldMatrix<T, n, m>, A>>
        {
            /** @brief The type of the domain of the solver */
            using domain_type = BlockVector<
                FieldVector<T, m>,
                typename std::allocator_traits<A>::template rebind_alloc<FieldVector<T, m>>>;
            /** @brief The type of the range of the solver */
            using range_type = BlockVector<
                FieldVector<T, n>,
                typename std::allocator_traits<A>::template rebind_alloc<FieldVector<T, n>>>;
        };

        template <typename T, typename A>
        struct UMFPackVectorChooser<BCRSMatrix<T, A>>
        {
            /** @brief The type of the domain of the solver */
            using domain_type = BlockVector<T, A>;
            /** @brief The type of the range of the solver */
            using range_type = BlockVector<T, A>;
        };
    }

    /** @brief The %UMFPackMOES direct sparse solver
   *
   * Details on UMFPackMOES can be found on
   * http://www.cise.ufl.edu/research/sparse/umfpack/
   *
   * %UMFPackMOES will always use double precision.
   * For complex matrices use a matrix type with std::complex<double>
   * as the underlying number type.
   *
   * \tparam Matrix the matrix type defining the system
   *
   * \note This will only work if dune-istl has been configured to use UMFPackMOES
   */
    template <typename M>
    class UMFPackMOES
        : public InverseOperator<
              typename Impl::UMFPackVectorChooser<M>::domain_type,
              typename Impl::UMFPackVectorChooser<M>::range_type>
    {
        using T = typename M::field_type;

    public:
        /** @brief The matrix type. */
        using Matrix = M;
        using matrix_type = M;
        /** @brief The corresponding UMFPackMOES matrix type.*/
        typedef ISTL::Impl::BCCSMatrix<typename Matrix::field_type, long int> UMFPackMatrix;
        /** @brief Type of an associated initializer class. */
        typedef ISTL::Impl::BCCSMatrixInitializer<M, long int> MatrixInitializer;
        /** @brief The type of the domain of the solver. */
        using domain_type = typename Impl::UMFPackVectorChooser<M>::domain_type;
        /** @brief The type of the range of the solver. */
        using range_type = typename Impl::UMFPackVectorChooser<M>::range_type;

        //! Category of the solver (see SolverCategory::Category)
        virtual SolverCategory::Category category() const
        {
            return SolverCategory::Category::sequential;
        }

        /** @brief Construct a solver object from a matrix
     *
     * This computes the matrix decomposition, and may take a long time
     * (and use a lot of memory).
     *
     *  @param matrix the matrix to solve for
     *  @param verbose [0..2] set the verbosity level, defaults to 0
     */
        UMFPackMOES(const Matrix &matrix, int verbose = 0) : matrixIsLoaded_(false)
        {
            //check whether T is a supported type
            static_assert((std::is_same<T, double>::value) || (std::is_same<T, std::complex<double>>::value),
                          "Unsupported Type in UMFPackMOES (only double and std::complex<double> supported)");
            Caller::defaults(UMF_Control);
            setVerbosity(verbose);
            setMatrix(matrix);
        }

        /** @brief Constructor for compatibility with SuperLU standard constructor
     *
     * This computes the matrix decomposition, and may take a long time
     * (and use a lot of memory).
     *
     * @param matrix the matrix to solve for
     * @param verbose [0..2] set the verbosity level, defaults to 0
     */
        UMFPackMOES(const Matrix &matrix, int verbose, bool) : matrixIsLoaded_(false)
        {
            //check whether T is a supported type
            static_assert((std::is_same<T, double>::value) || (std::is_same<T, std::complex<double>>::value),
                          "Unsupported Type in UMFPackMOES (only double and std::complex<double> supported)");
            Caller::defaults(UMF_Control);
            setVerbosity(verbose);
            setMatrix(matrix);
        }

        /** @brief Construct a solver object from a matrix
     *
     * @param matrix  the matrix to solve for
     * @param config  ParameterTree containing solver parameters.
     *
     * ParameterTree Key | Meaning
     * ------------------|------------
     * verbose           | The verbosity level. default=0
    */
        UMFPackMOES(const Matrix &mat_, const ParameterTree &config)
            : UMFPackMOES(mat_, config.get<int>("verbose", 0))
        {
        }

        /** @brief default constructor
     */
        UMFPackMOES() : matrixIsLoaded_(false), verbosity_(0)
        {
            //check whether T is a supported type
            static_assert((std::is_same<T, double>::value) || (std::is_same<T, std::complex<double>>::value),
                          "Unsupported Type in UMFPackMOES (only double and std::complex<double> supported)");
            Caller::defaults(UMF_Control);
        }

        /** @brief Try loading a decomposition from file and do a decomposition if unsuccessful
     * @param mat_ the matrix to decompose when no decoposition file found
     * @param file the decomposition file
     * @param verbose the verbosity level
     *
     * Use saveDecomposition(char* file) for manually storing a decomposition. This constructor
     * will decompose mat_ and store the result to file if no file wasn't found in the first place.
     * Thus, if you always use this you will only compute the decomposition once (and when you manually
     * deleted the decomposition file).
     */
        UMFPackMOES(const Matrix &mat_, const char *file, int verbose = 0)
        {
            //check whether T is a supported type
            static_assert((std::is_same<T, double>::value) || (std::is_same<T, std::complex<double>>::value),
                          "Unsupported Type in UMFPackMOES (only double and std::complex<double> supported)");
            Caller::defaults(UMF_Control);
            setVerbosity(verbose);
            int errcode = Caller::load_numeric(&UMF_Numeric, const_cast<char *>(file));
            if ((errcode == UMFPACK_ERROR_out_of_memory) || (errcode == UMFPACK_ERROR_file_IO))
            {
                matrixIsLoaded_ = false;
                setMatrix(mat_);
                saveDecomposition(file);
            }
            else
            {
                matrixIsLoaded_ = true;
                std::cout << "UMFPackMOES decomposition successfully loaded from " << file << std::endl;
            }
        }

        /** @brief try loading a decomposition from file
     * @param file the decomposition file
     * @param verbose the verbosity level
     * @throws Dune::Exception When not being able to load the file. Does not need knowledge of the
     * actual matrix!
     */
        UMFPackMOES(const char *file, int verbose = 0)
        {
            //check whether T is a supported type
            static_assert((std::is_same<T, double>::value) || (std::is_same<T, std::complex<double>>::value),
                          "Unsupported Type in UMFPackMOES (only double and std::complex<double> supported)");
            Caller::defaults(UMF_Control);
            int errcode = Caller::load_numeric(&UMF_Numeric, const_cast<char *>(file));
            if (errcode == UMFPACK_ERROR_out_of_memory)
                DUNE_THROW(Dune::Exception, "ran out of memory while loading UMFPackMOES decomposition");
            if (errcode == UMFPACK_ERROR_file_IO)
                DUNE_THROW(Dune::Exception, "IO error while loading UMFPackMOES decomposition");
            matrixIsLoaded_ = true;
            std::cout << "UMFPackMOES decomposition successfully loaded from " << file << std::endl;
            setVerbosity(verbose);
        }

        virtual ~UMFPackMOES()
        {
            if ((umfpackMatrix_.N() + umfpackMatrix_.M() > 0) || matrixIsLoaded_)
                free();
        }

        /**
     *  \copydoc InverseOperator::apply(X&, Y&, InverseOperatorResult&)
     */
        virtual void apply(domain_type &x, range_type &b, InverseOperatorResult &res)
        {
            if (umfpackMatrix_.N() != b.dim())
                DUNE_THROW(Dune::ISTLError, "Size of right-hand-side vector b does not match the number of matrix rows!");
            if (umfpackMatrix_.M() != x.dim())
                DUNE_THROW(Dune::ISTLError, "Size of solution vector x does not match the number of matrix columns!");
            if (b.size() == 0)
                return;

            double UMF_Apply_Info[UMFPACK_INFO];
            Caller::solve(UMFPACK_A,
                          umfpackMatrix_.getColStart(),
                          umfpackMatrix_.getRowIndex(),
                          umfpackMatrix_.getValues(),
                          reinterpret_cast<double *>(&x[0]),
                          reinterpret_cast<double *>(&b[0]),
                          UMF_Numeric,
                          UMF_Control,
                          UMF_Apply_Info);

            //this is a direct solver
            res.iterations = 1;
            res.converged = true;
            res.elapsed = UMF_Apply_Info[UMFPACK_SOLVE_WALLTIME];

            printOnApply(UMF_Apply_Info);
        }

        /**
     *  \copydoc InverseOperator::apply(X&,Y&,double,InverseOperatorResult&)
     */
        virtual void apply(domain_type &x, range_type &b, [[maybe_unused]] double reduction, InverseOperatorResult &res)
        {
            apply(x, b, res);
        }

        /**
     * @brief additional apply method with c-arrays in analogy to superlu
     * @param x solution array
     * @param b rhs array
     */
        // TODO: This is where I have to write my own solver
        void apply(T *x, T *b)
        {
            double UMF_Apply_Info[UMFPACK_INFO];
            Caller::solve(UMFPACK_A,
                          umfpackMatrix_.getColStart(),
                          umfpackMatrix_.getRowIndex(),
                          umfpackMatrix_.getValues(),
                          x,
                          b,
                          UMF_Numeric,
                          UMF_Control,
                          UMF_Apply_Info);
            printOnApply(UMF_Apply_Info);
        }

        /** 
         * @brief Ax = b solved via LU decomposition for use with moes own multivectors
         * @param b rhs Multivector
         * @param x solution Multivector
        */
        void moesInversePowerIteration(const std::shared_ptr<double[]> &b, std::shared_ptr<double[]> &x, const size_t N, const size_t rhsWidth)
        {

            // do_recip is a return value that tells me if the values in the diagonal matrix R are divided by or multiplied by

            size_t qCols = rhsWidth / 8;
            std::shared_ptr<double[]> xTmp(new double[N * rhsWidth]);
            //std::shared_ptr<double[]> xTmp(new double[N * rhsWidth]); // since b needs to stay constant and we need to store intermediate results
            /*
            
            // L is n_row x min(n_row, n_col)
            // U is min(n_row, n_col) x n_col
            //  PAQ=LU, PRAQ=LU, or P(R\A)Q=LU=PR^{-1} A Q
            // Since the arrays above stay the same for the power iteration, I should probably do the power iteration below, or move this stuff to the initialization
            // Here I have to put in my solver, i.e., solve PRAQ = LU and Ax = b
            // Note: P^-1 = P^T (property of the permutation matrix)
            // Therefore: A = R^{-1} P^{-1} LU Q^{-1}. We can calculate the subproblems as follows:
            
            */
            // t = PRb with t = LU Q^{-1} x
            // here: x = t, b = b

            applyScalingP(b, x, Rs, P, N, qCols, do_recip);
            // solve Lu = t with u = U Q^{-1} x
            // here: xTmp = t, x = u
            solveL(x, xTmp, Lp, Lj, Lx, n_row, qCols);
            // U x = Q u
            // here: x = x, xTmp = u
            solveUQ(xTmp, x, Q, Up, Ui, Ux, n_col, qCols);
            // Normalize x
            normalize(x, N, qCols);
        }

        double getFlops() { return flops; }
        double getFtime() { return ftime; }
        double getCondNumber() { return conditionNumberEstimate; }
        double getNonZeroL() { return nonZeroL; }
        double getNonZeroU() { return nonZeroU; }

        /** @brief Set UMFPackMOES-specific options
     *
     * This method allows to set various options that control the UMFPackMOES solver.
     * More specifically, it allows to set values in the UMF_Control array.
     * Please see the UMFPackMOES documentation for a list of possible options and values.
     *
     * \param option Entry in the UMF_Control array, e.g., UMFPACK_IRSTEP
     * \param value Corresponding value
     *
     * \throws RangeError If nonexisting option was requested
     */
        void setOption(unsigned int option, double value)
        {
            if (option >= UMFPACK_CONTROL)
                DUNE_THROW(RangeError, "Requested non-existing UMFPackMOES option");

            UMF_Control[option] = value;
        }

        /** @brief saves a decomposition to a file
     * @param file the filename to save to
     */
        void saveDecomposition(const char *file)
        {
            int errcode = Caller::save_numeric(UMF_Numeric, const_cast<char *>(file));
            if (errcode != UMFPACK_OK)
                DUNE_THROW(Dune::Exception, "IO ERROR while trying to save UMFPackMOES decomposition");
        }

        /** @brief Initialize data from given matrix. */
        void setMatrix(const Matrix &matrix)
        {
            if ((umfpackMatrix_.N() + umfpackMatrix_.M() > 0) || matrixIsLoaded_)
                free();
            if (matrix.N() == 0 or matrix.M() == 0)
                return;

            if (umfpackMatrix_.N() + umfpackMatrix_.M() + umfpackMatrix_.nonzeroes() != 0)
                umfpackMatrix_.free();
            umfpackMatrix_.setSize(MatrixDimension<Matrix>::rowdim(matrix),
                                   MatrixDimension<Matrix>::coldim(matrix));
            ISTL::Impl::BCCSMatrixInitializer<Matrix, long int> initializer(umfpackMatrix_);

            copyToBCCSMatrix(initializer, matrix);

            decompose();
        }

        template <class S>
        void setSubMatrix(const Matrix &_mat, const S &rowIndexSet)
        {
            if ((umfpackMatrix_.N() + umfpackMatrix_.M() > 0) || matrixIsLoaded_)
                free();

            if (umfpackMatrix_.N() + umfpackMatrix_.M() + umfpackMatrix_.nonzeroes() != 0)
                umfpackMatrix_.free();

            umfpackMatrix_.setSize(rowIndexSet.size() * MatrixDimension<Matrix>::rowdim(_mat) / _mat.N(),
                                   rowIndexSet.size() * MatrixDimension<Matrix>::coldim(_mat) / _mat.M());
            ISTL::Impl::BCCSMatrixInitializer<Matrix, long int> initializer(umfpackMatrix_);

            copyToBCCSMatrix(initializer, ISTL::Impl::MatrixRowSubset<Matrix, std::set<std::size_t>>(_mat, rowIndexSet));

            decompose();
        }

        /** @brief sets the verbosity level for the UMFPackMOES solver
     * @param v verbosity level
     * The following levels are implemented:
     * 0 - only error messages
     * 1 - a bit of statistics on decomposition and solution
     * 2 - lots of statistics on decomposition and solution
     */
        void setVerbosity(int v)
        {
            verbosity_ = v;
            // set the verbosity level in UMFPackMOES
            if (verbosity_ == 0)
                UMF_Control[UMFPACK_PRL] = 1;
            if (verbosity_ == 1)
                UMF_Control[UMFPACK_PRL] = 2;
            if (verbosity_ == 2)
                UMF_Control[UMFPACK_PRL] = 4;
        }

        /**
     * @brief Return the matrix factorization.
     * @warning It is up to the user to keep consistency.
     */
        void *getFactorization()
        {
            return UMF_Numeric;
        }

        /**
     * @brief Return the column compress matrix from UMFPackMOES.
     * @warning It is up to the user to keep consistency.
     */
        UMFPackMatrix &getInternalMatrix()
        {
            return umfpackMatrix_;
        }

        /**
     * @brief free allocated space.
     * @warning later calling apply will result in an error.
     */
        void free()
        {
            if (!matrixIsLoaded_)
            {
                Caller::free_symbolic(&UMF_Symbolic);
                umfpackMatrix_.free();
            }
            Caller::free_numeric(&UMF_Numeric);
            matrixIsLoaded_ = false;
            delete[] Lp;
            delete[] Lj;
            delete[] Lx;
            delete[] Up;
            delete[] Ui;
            delete[] Ux;
            delete[] P;
            delete[] Q;
            delete[] Dx;
            delete[] Rs;
        }

        const char *name() { return "UMFPACK"; }

    private:
        typedef typename Dune::UMFPackMethodChooser<T> Caller;

        template <class Mat, class X, class TM, class TD, class T1>
        friend class SeqOverlappingSchwarz;
        friend struct SeqOverlappingSchwarzAssemblerHelper<UMFPackMOES<Matrix>, true>;

        /** @brief computes the LU Decomposition */
        void decompose()
        {
            double UMF_Decomposition_Info[UMFPACK_INFO];
            Caller::symbolic(static_cast<int>(umfpackMatrix_.N()),
                             static_cast<int>(umfpackMatrix_.N()),
                             umfpackMatrix_.getColStart(),
                             umfpackMatrix_.getRowIndex(),
                             reinterpret_cast<double *>(umfpackMatrix_.getValues()),
                             &UMF_Symbolic,
                             UMF_Control,
                             UMF_Decomposition_Info);
            Caller::numeric(umfpackMatrix_.getColStart(),
                            umfpackMatrix_.getRowIndex(),
                            reinterpret_cast<double *>(umfpackMatrix_.getValues()),
                            UMF_Symbolic,
                            &UMF_Numeric,
                            UMF_Control,
                            UMF_Decomposition_Info);
            Caller::report_status(UMF_Control, UMF_Decomposition_Info[UMFPACK_STATUS]);
            if (verbosity_ == 1)
            {
                std::cout << "[UMFPackMOES Decomposition]" << std::endl;
                std::cout << "Wallclock Time taken: " << UMF_Decomposition_Info[UMFPACK_NUMERIC_WALLTIME] << " (CPU Time: " << UMF_Decomposition_Info[UMFPACK_NUMERIC_TIME] << ")" << std::endl;
                std::cout << "Flops taken: " << UMF_Decomposition_Info[UMFPACK_FLOPS] << std::endl;
                std::cout << "Peak Memory Usage: " << UMF_Decomposition_Info[UMFPACK_PEAK_MEMORY] * UMF_Decomposition_Info[UMFPACK_SIZE_OF_UNIT] << " bytes" << std::endl;
                std::cout << "Condition number estimate: " << 1. / UMF_Decomposition_Info[UMFPACK_RCOND] << std::endl;
                std::cout << "Numbers of non-zeroes in decomposition: L: " << UMF_Decomposition_Info[UMFPACK_LNZ] << " U: " << UMF_Decomposition_Info[UMFPACK_UNZ] << std::endl;
            }
            if (verbosity_ == 2)
            {
                Caller::report_info(UMF_Control, UMF_Decomposition_Info);
            }
            int status;

            status = Caller::get_lunz(&lnz, &unz, &n_row, &n_col, &nz_udiag,
                                      UMF_Numeric);
            Lp = new int64_t[n_row + 1];
            Lj = new int64_t[lnz];
            Lx = new double[lnz];
            Up = new int64_t[n_col + 1];
            Ui = new int64_t[unz];
            Ux = new double[unz];
            P = new int64_t[n_row];
            Q = new int64_t[n_col];
            Dx = new double[std::min(n_row, n_col)];
            Rs = new double[n_row];
            flops = UMF_Decomposition_Info[UMFPACK_FLOPS];
            ftime = UMF_Decomposition_Info[UMFPACK_NUMERIC_WALLTIME];
            conditionNumberEstimate = 1. / UMF_Decomposition_Info[UMFPACK_RCOND];
            nonZeroL = UMF_Decomposition_Info[UMFPACK_LNZ];
            nonZeroU = UMF_Decomposition_Info[UMFPACK_UNZ];
            status = Caller::get_numeric(Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx,
                                         &do_recip, Rs, UMF_Numeric);
        }

        void printOnApply(double *UMF_Info)
        {
            Caller::report_status(UMF_Control, UMF_Info[UMFPACK_STATUS]);
            if (verbosity_ > 0)
            {
                std::cout << "[UMFPackMOES Solve]" << std::endl;
                std::cout << "Wallclock Time: " << UMF_Info[UMFPACK_SOLVE_WALLTIME] << " (CPU Time: " << UMF_Info[UMFPACK_SOLVE_TIME] << ")" << std::endl;
                std::cout << "Flops Taken: " << UMF_Info[UMFPACK_SOLVE_FLOPS] << std::endl;
                std::cout << "Iterative Refinement steps taken: " << UMF_Info[UMFPACK_IR_TAKEN] << std::endl;
                std::cout << "Error Estimate: " << UMF_Info[UMFPACK_OMEGA1] << " resp. " << UMF_Info[UMFPACK_OMEGA2] << std::endl;
            }
        }

        UMFPackMatrix umfpackMatrix_;
        bool matrixIsLoaded_;
        int verbosity_;
        void *UMF_Symbolic;
        void *UMF_Numeric;
        int64_t lnz, unz, n_row, n_col, nz_udiag, do_recip;
        int64_t *Lp, *Lj, *Up, *Ui, *P, *Q;
        double *Lx, *Ux, *Dx, *Rs;
        double UMF_Control[UMFPACK_CONTROL];
        double flops, ftime, conditionNumberEstimate, nonZeroL, nonZeroU;
        // Add the proper things here
    };

    template <typename T, typename A, int n, int m>
    struct IsDirectSolver<UMFPackMOES<BCRSMatrix<FieldMatrix<T, n, m>, A>>>
    {
        enum
        {
            value = true
        };
    };

    template <typename T, typename A>
    struct StoresColumnCompressed<UMFPackMOES<BCRSMatrix<T, A>>>
    {
        enum
        {
            value = true
        };
    };

    struct UMFPackCreator
    {
        template <class F, class = void>
        struct isValidBlock : std::false_type
        {
        };
        template <class B>
        struct isValidBlock<B, std::enable_if_t<std::is_same<typename FieldTraits<B>::real_type, double>::value>> : std::true_type
        {
        };

        template <typename TL, typename M>
        std::shared_ptr<Dune::InverseOperator<typename Dune::TypeListElement<1, TL>::type,
                                              typename Dune::TypeListElement<2, TL>::type>>
        operator()(TL /*tl*/, const M &mat, const Dune::ParameterTree &config,
                   std::enable_if_t<
                       isValidBlock<typename Dune::TypeListElement<1, TL>::type::block_type>::value, int> = 0) const
        {
            int verbose = config.get("verbose", 0);
            return std::make_shared<Dune::UMFPackMOES<M>>(mat, verbose);
        }

        // second version with SFINAE to validate the template parameters of UMFPackMOES
        template <typename TL, typename M>
        std::shared_ptr<Dune::InverseOperator<typename Dune::TypeListElement<1, TL>::type,
                                              typename Dune::TypeListElement<2, TL>::type>>
        operator()(TL /*tl*/, const M & /*mat*/, const Dune::ParameterTree & /*config*/,
                   std::enable_if_t<
                       !isValidBlock<typename Dune::TypeListElement<1, TL>::type::block_type>::value, int> = 0) const
        {
            DUNE_THROW(UnsupportedType,
                       "Unsupported Type in UMFPackMOES (only double and std::complex<double> supported)");
        }
    };
    DUNE_REGISTER_DIRECT_SOLVER("umfpack", Dune::UMFPackCreator());
} // end namespace Dune

//#endif // HAVE_SUITESPARSE_UMFPACK
#endif //  // end namespace Dun    namespace Dune
