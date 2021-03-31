use std::fmt;
use std::io::{Cursor, Read, Write};

#[cfg(feature = "blst")]
use blstrs::{
    Bls12, Fp12 as Fq12, Compress, Engine, G1Affine, G1Projective, G2Affine, G2Projective,
    PairingCurveAffine,
};
#[cfg(feature = "pairing")]
use paired::{
    bls12_381::{Bls12, Fq12, G1Affine, G2Affine, G1 as G1Projective, G2 as G2Projective},
    Compress, Engine, PairingCurveAffine
};
use groupy::{CurveAffine, CurveProjective, EncodedPoint};
use rand_core::{RngCore, SeedableRng};

#[derive(PartialEq, Clone, Debug)]
struct TestVector {
    g1: Vec<u8>,
    g2: Vec<u8>,
    gt: Vec<u8>,
}

impl fmt::Display for TestVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?}, {:?}, {:?})", self.g1, self.g2, self.gt)
    }
}

fn get_test_vectors() -> Vec<TestVector> {
    let blst_test_vector = TestVector {
        g1: vec![185, 247, 28, 228, 44, 194, 179, 143, 89, 126, 183, 34, 5, 199, 81, 133, 20, 155, 254, 35, 226, 12, 237, 15, 243, 203, 151, 199, 233, 73, 68, 199, 47, 189, 146, 181, 176, 142, 183, 109, 250, 26, 152, 254, 124, 240, 206, 126],
        g2: vec![184, 182, 217, 215, 120, 52, 243, 127, 253, 223, 189, 8, 108, 94, 168, 77, 30, 138, 217, 1, 111, 98, 239, 103, 111, 183, 239, 125, 122, 241, 74, 44, 237, 194, 202, 181, 116, 247, 152, 98, 248, 90, 243, 44, 189, 175, 153, 128, 2, 154, 186, 54, 183, 234, 215, 121, 128, 144, 0, 58, 120, 163, 105, 241, 113, 241, 14, 166, 67, 82, 254, 61, 167, 101, 57, 25, 177, 9, 62, 117, 27, 162, 94, 242, 82, 213, 38, 174, 246, 120, 120, 111, 36, 83, 202, 111],
        gt: vec![23, 85, 39, 132, 23, 232, 52, 34, 157, 111, 81, 149, 115, 162, 96, 179, 178, 142, 95, 241, 70, 36, 12, 33, 204, 207, 26, 136, 153, 120, 211, 255, 20, 31, 146, 37, 115, 141, 12, 30, 168, 198, 88, 213, 187, 70, 215, 1, 208, 187, 132, 42, 102, 235, 75, 103, 216, 60, 36, 228, 102, 42, 26, 104, 57, 118, 31, 15, 148, 42, 170, 188, 84, 91, 173, 148, 209, 128, 4, 196, 122, 183, 19, 5, 32, 236, 215, 213, 30, 246, 167, 87, 53, 74, 136, 10, 248, 40, 192, 76, 185, 205, 13, 237, 165, 75, 172, 146, 86, 0, 84, 245, 238, 90, 131, 231, 116, 61, 52, 136, 60, 146, 131, 239, 72, 186, 128, 139, 239, 67, 40, 106, 20, 13, 127, 104, 171, 79, 251, 106, 135, 158, 128, 24, 13, 151, 242, 246, 172, 238, 84, 179, 223, 78, 68, 144, 147, 200, 18, 87, 103, 205, 217, 81, 250, 236, 74, 30, 207, 89, 155, 39, 247, 132, 252, 129, 192, 97, 122, 196, 103, 234, 141, 22, 143, 215, 183, 244, 117, 119, 253, 17, 226, 120, 123, 122, 63, 1, 240, 144, 26, 143, 203, 187, 158, 98, 88, 209, 87, 237, 43, 248, 79, 101, 247, 240, 22, 110, 50, 69, 58, 90, 109, 108, 61, 112, 175, 57, 236, 143, 169, 130, 89, 122, 187, 229, 160, 254, 146, 2, 157, 188, 81, 158, 144, 59, 4, 105, 27, 139, 43, 21, 216, 166, 119, 37, 193, 126, 97, 34, 5, 54, 178, 122, 84, 103, 209, 76, 15, 165, 53, 101, 56, 238, 66, 149, 68, 192, 146, 69, 54, 60, 255, 253, 179, 169, 244, 11],
    };

    let pairing_test_vector = TestVector {
        g1: vec![182, 168, 107, 123, 173, 195, 197, 50, 177, 172, 230, 210, 37, 224, 168, 4, 37, 221, 248, 30, 230, 150, 241, 192, 63, 230, 107, 221, 88, 241, 241, 167, 135, 229, 254, 64, 124, 131, 143, 173, 149, 114, 31, 94, 59, 138, 177, 210],
        g2: vec![178, 125, 150, 185, 199, 227, 175, 20, 104, 17, 205, 149, 8, 239, 193, 165, 181, 41, 1, 117, 51, 121, 127, 250, 192, 110, 187, 206, 71, 54, 122, 221, 234, 203, 68, 201, 29, 131, 121, 143, 36, 224, 100, 10, 39, 204, 106, 18, 18, 66, 12, 88, 166, 229, 237, 217, 200, 234, 9, 85, 223, 124, 143, 133, 130, 234, 155, 123, 174, 28, 153, 131, 95, 240, 76, 137, 40, 135, 70, 89, 145, 248, 182, 213, 156, 6, 17, 174, 254, 63, 200, 67, 34, 254, 244, 88],
        gt: vec![19, 229, 127, 223, 132, 140, 57, 63, 37, 44, 59, 77, 75, 37, 73, 151, 126, 232, 202, 245, 160, 45, 152, 131, 218, 52, 99, 73, 122, 2, 121, 213, 37, 233, 20, 82, 222, 17, 65, 6, 58, 33, 44, 45, 123, 48, 145, 11, 236, 192, 31, 151, 229, 1, 94, 179, 188, 6, 36, 205, 0, 231, 83, 170, 113, 187, 227, 76, 217, 166, 9, 175, 26, 66, 15, 164, 244, 84, 130, 58, 37, 237, 108, 131, 240, 58, 127, 230, 193, 4, 251, 138, 76, 8, 95, 21, 137, 63, 130, 31, 212, 22, 20, 122, 95, 158, 203, 36, 43, 21, 45, 88, 20, 179, 220, 57, 154, 165, 42, 195, 47, 29, 232, 50, 102, 253, 200, 29, 219, 58, 202, 28, 54, 100, 73, 156, 236, 88, 205, 211, 231, 99, 26, 8, 59, 110, 83, 165, 25, 45, 225, 138, 182, 149, 126, 146, 187, 251, 116, 203, 12, 242, 230, 217, 242, 216, 123, 144, 131, 17, 191, 163, 101, 23, 89, 10, 22, 244, 58, 251, 65, 164, 3, 228, 120, 60, 128, 70, 248, 126, 143, 23, 212, 148, 68, 216, 24, 206, 7, 171, 240, 132, 47, 23, 114, 200, 38, 77, 244, 120, 215, 112, 89, 183, 42, 102, 177, 197, 146, 14, 142, 100, 230, 140, 213, 124, 204, 113, 52, 159, 222, 241, 89, 161, 170, 74, 70, 142, 250, 4, 118, 26, 202, 7, 92, 246, 1, 74, 134, 157, 197, 199, 125, 178, 128, 235, 46, 193, 57, 70, 25, 104, 199, 100, 234, 84, 207, 72, 184, 46, 64, 117, 44, 73, 150, 149, 138, 61, 100, 143, 198, 70, 144, 168, 90, 241, 5, 7],
    };

    vec![blst_test_vector, pairing_test_vector]
}

fn compute_test_vector<R: RngCore>(r: &mut R) -> TestVector {
    let g1: G1Affine = G1Projective::random(r).into();
    let g2: G2Affine = G2Projective::random(r).into();
    let gt =
        Bls12::final_exponentiation(&Bls12::miller_loop(&[(&g1.prepare(), &g2.prepare())]))
            .unwrap();
    let mut eg1: Vec<u8> = Vec::new();
    eg1.write_all(g1.into_compressed().as_ref()).unwrap();
    let mut eg2: Vec<u8> = Vec::new();
    eg2.write_all(g2.into_compressed().as_ref()).unwrap();
    let mut egt: Vec<u8> = Vec::new();
    gt.write_compressed(&mut egt).unwrap();
    let vector = TestVector {
        g1: eg1,
        g2: eg2,
        gt: egt,
    };

    vector
}

fn verify_test_vector(t: TestVector) {
    let mut g1c = <G1Affine as CurveAffine>::Compressed::empty();
    Cursor::new(t.g1).read_exact(g1c.as_mut()).unwrap();
    let g1 = g1c.into_affine().unwrap();

    let mut g2c = <G2Affine as CurveAffine>::Compressed::empty();
    Cursor::new(t.g2).read_exact(g2c.as_mut()).unwrap();
    let g2 = g2c.into_affine().unwrap();

    let gtexp =
        Bls12::final_exponentiation(&Bls12::miller_loop(&[(&g1.prepare(), &g2.prepare())]))
            .unwrap();
    let gt = <Fq12 as Compress>::read_compressed(&mut Cursor::new(t.gt)).unwrap();
    assert_eq!(gtexp, gt);
}

#[test]
fn test_compat() {
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(0u64);

    // Generate and verify a test vector using the current backend
    let v1 = compute_test_vector(&mut rng);
    verify_test_vector(v1);

    // Retrieve test vectors generated with each backend and verify
    // them using the current backend
    let vecs = get_test_vectors();
    for vec in vecs {
        verify_test_vector(vec);
    }
}
