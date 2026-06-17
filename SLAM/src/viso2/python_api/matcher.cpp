#include <boost/cstdint.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <fstream>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

#include <viso2/matcher.h>

using namespace viso2;

namespace p = boost::python;
namespace np = p::numpy;

// Converts a C++ vector to a python list
// // http://stackoverflow.com/questions/5314319/how-to-export-stdvector
template <class T>
struct VectorToListConverter
{
    static PyObject *convert(const std::vector<T> &vector)
    {
        boost::python::list *l = new boost::python::list();
        for (const auto &el : vector)
        {
            l->append(el);
	}
	return l->ptr();
    }
};
// // tell the vector indexing suite not to use operator == since undefined
//    namespace boost { namespace python{namespace indexing {
//    template<>
//    struct value_traits<Matcher::p_match> : public value_traits<int>
//    {
//        static bool const equality_comparable = false;
//        static bool const lessthan_comparable = false;
//    };
//    }}}

///@brief Wrap the push back call of matcher.
///       It requires that the passed NumPy array be exactly what we're 
///       looking for - no conversion from nested sequences or arrays with 
///       other data types, because we want to modify it in-place. 
///       Modified example from boost_1_63_0/libs/python/example/numpy/wrap.cpp
inline void wrapMatcherPushBack(Matcher &obj, np::ndarray const &array, bool replace)
{
    if (array.get_dtype() != np::dtype::get_builtin<uint8_t>())
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        p::throw_error_already_set();
    }
    if (array.get_nd() != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        p::throw_error_already_set();
    }

    int32_t height = array.shape(0);
    int32_t width = array.shape(1);
    int32_t dims[] = {width, height, width};

    obj.pushBack(reinterpret_cast<uint8_t *>(array.get_data()), dims, replace);
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(matchFeatures_overload, Matcher::matchFeatures, 1, 2)

BOOST_PYTHON_MODULE(matcher)
{
    np::initialize(); // have to put this in any module that uses Boost.NumPy

    p::class_<Matcher::parameters>("MatcherParams")
        .def_readwrite("nms_n", &Matcher::parameters::nms_n)
        .def_readwrite("nms_tau", &Matcher::parameters::nms_tau)
        .def_readwrite("match_binsize", &Matcher::parameters::match_binsize)
        .def_readwrite("match_radius", &Matcher::parameters::match_radius)
        .def_readwrite("match_disp_tolerance", &Matcher::parameters::match_disp_tolerance)
        .def_readwrite("outlier_disp_tolerance", &Matcher::parameters::outlier_disp_tolerance)
        .def_readwrite("outlier_flow_tolerance", &Matcher::parameters::outlier_flow_tolerance)
        .def_readwrite("multi_stage", &Matcher::parameters::multi_stage)
        .def_readwrite("half_resolution", &Matcher::parameters::half_resolution)
        .def_readwrite("refinement", &Matcher::parameters::refinement)
        .def_readwrite("f", &Matcher::parameters::f)
        .def_readwrite("cu", &Matcher::parameters::cu)
        .def_readwrite("cv", &Matcher::parameters::cv)
        .def_readwrite("base", &Matcher::parameters::base);

    p::class_<Matcher::p_match>("Match")
        .def_readwrite("u1p", &Matcher::p_match::u1p)
        .def_readwrite("v1p", &Matcher::p_match::v1p)
        .def_readwrite("i1p", &Matcher::p_match::i1p)
        .def_readwrite("u1c", &Matcher::p_match::u1c)
        .def_readwrite("v1c", &Matcher::p_match::v1c)
        .def_readwrite("i1c", &Matcher::p_match::i1c)
        .def_readwrite("u2p", &Matcher::p_match::u2p)
        .def_readwrite("v2p", &Matcher::p_match::v2p)
        .def_readwrite("i2p", &Matcher::p_match::i2p)
        .def_readwrite("u2c", &Matcher::p_match::u2c)
        .def_readwrite("v2c", &Matcher::p_match::v2c)
        .def_readwrite("i2c", &Matcher::p_match::i2c);

    using Matches = std::vector<Matcher::p_match>;
//     p::class_<Matches>("Matches").def(p::vector_indexing_suite<Matches>());
    p::to_python_converter<Matches, VectorToListConverter<Matcher::p_match>>();

    p::class_<Matcher, boost::noncopyable>("Matcher", p::init<Matcher::parameters>())
        .def("getMatches", &Matcher::getMatches)
        .def("matchFeatures", &Matcher::matchFeatures, matchFeatures_overload());

    p::def("pushBack", &wrapMatcherPushBack);
}

