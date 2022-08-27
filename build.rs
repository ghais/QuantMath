fn main() {
    cc::Build::new()
        .file("lbr/src/lets_be_rational.cpp")
        .file("lbr/src/normaldistribution.cpp")
        .file("lbr/src/rationalcubic.cpp")
        .file("lbr/src/erf_cody.cpp")
        .include("lbr/src")
        .compile("lbr");
}