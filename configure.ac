AC_INIT([ssim],
	0.0.2)
dnl        m4_esyscmd([./git-version-gen .tarball-version]),
dnl        [bug@example])
AM_INIT_AUTOMAKE(foreign dist-xz no-dist-gzip)
AC_CONFIG_SRCDIR(ssim.cpp)
dnl AC_ENABLE_SHARED
dnl AC_DISABLE_STATIC
dnl AC_PROG_LIBTOOL(libtool)
AC_CONFIG_MACRO_DIR([m4])
AC_PROG_CC
AC_PROG_CXX
AC_PROG_RANLIB
dnl AC_CONFIG_HEADER(src/config.h)
dnl AC_PROG_LEX
dnl AC_PROG_YACC
dnl AC_CHECK_HEADERS(unistd.h)
AC_CHECK_SIZEOF(int *)
dnl AC_EXEEXT
AC_LANG_CPLUSPLUS
AC_C_BIGENDIAN

AC_ARG_ENABLE(debug,
[  --enable-debug    Turn on debug info.],
[case "${enableval}" in
  yes) debug=true ;;
  no)  debug=false ;;
  *) AC_MSG_ERROR(bad value ${enableval} for --enable-debug) ;;
esac],[debug=true])
AM_CONDITIONAL(NDEBUG, test x$debug = xfalse)

dnl files to be generated
AC_OUTPUT(Makefile)
