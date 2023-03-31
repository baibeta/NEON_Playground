#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void neon_add(float32_t *a, float32_t *b, float32_t *result, int n) {
    int i;
    float32x4_t va, vb, vr;

    for (i = 0; i < n; i += 4) {
        va = vld1q_f32(&a[i]);
        vb = vld1q_f32(&b[i]);
        vr = vaddq_f32(va, vb);
        vst1q_f32(&result[i], vr);

        if (n - i < 4) {
            for (int j = n - n % 4; j < n; j++) {
                result[j] = a[j] + b[j];
                }
        }
    }
}


void load_lane_1() {
    int8_t my_array[4] = {1,5,8,10};

    int8x8_t result;
    result = vld1_lane_s8(my_array, result, 0);
    result = vld1_lane_s8(my_array+1, result, 1);
    result = vld1_lane_s8(my_array+2, result, 2);
    result = vld1_lane_s8(my_array+3, result, 3);

    // Result: [1 5 8 10]
    printf("Result: [%d %d %d %d]\n", 
           result[0], result[1], result[2], result[3]);
}


void test_complex() {
    float a[2] = {12, 34};
    float b[2] = {56, 78};
    float32x2_t va, vb, vc;


    va = vld1_f32(a);
    vb = vld1_f32(b);
    // 复数旋转90°加
    // 看上去是先把b向量旋转90度，即(x,y)->(-y,x)，然后再让a+b
    // a[0]-b[1], a[1]+b[0]
    vc = vcadd_rot90_f32(va,vb);
    // -66.000000 90.000000 
    printf("complex add result = %f %f \n",vc[0], vc[1]);


    // 复数旋转270°加
    // 看上去是先把b向量旋转270度，即(x,y)->(y,-x)，然后再让a+b
    // a[0]+b[1], a[1]-b[0]
    vc = vcadd_rot270_f32(va,vb);
    // 90.000000 -22.000000 
    printf("complex add result = %f %f \n",vc[0], vc[1]);

}





void test_complex_mac() {
    float a[2] = {12, 34};
    float b[2] = {56, 78};
    float c[2] = {111, 222};
    float32x2_t va, vb, vc;

    va = vld1_f32(a);
    vb = vld1_f32(b);
    vc = vld1_f32(c);

    // c1 + a1 * b1, c2 + a1 * b2
    vc = vcmla_f32(vc, va, vb);
    printf("complex mac result = %f %f \n",vc[0], vc[1]);

}

void test_complex_mac_270() {
    float a[2] = {12, 34};
    float b[2] = {56, 78};
    float c[2] = {111, 222};
    float32x2_t va, vb, vc;

    va = vld1_f32(a);
    vb = vld1_f32(b);
    vc = vld1_f32(c);

    // c1+ a2 * b2, c1 - a2 * b1
    // 111+34*78 222-34*56
    vc = vcmla_rot270_f32(vc, va, vb);
    printf("complex mac result = %f %f \n",vc[0], vc[1]);

}


void test_aes(){

    char* data = "abcdefghijklmnop";
    char* key = "1111111111111111";

    uint8x16_t v_data, v_key, v_aes_data;

    v_data = vld1q_u8(data);
    v_key = vld1q_u8(key);

    v_aes_data = vaeseq_u8(v_data, v_key);

    printf("AES Encrypted data = ");
    for(int i=0;i<16;i++) {
        printf("%c",v_aes_data[i]);
    }
    printf("\n");


    uint8x16_t v_aes_decryption = vaesdq_u8(v_aes_data,v_key);
    printf("AES Decrypted data = ");
    for(int i=0;i<16;i++) {
        printf("%c",v_aes_decryption[i]);
    }
    printf("\n");
}


void test_matrix() {
    // store in row
    int8_t a[16] = {1,2,3,4,5,6,7,8,
                        9,10,11,12,13,14,15,16};
    // store in columns
    int8_t b[16] = {17,18,19,20,21,22,23,24,
                        25,26,27,28,29,30,31,32};

    int8x16_t va,vb;

    va = vld1q_s8(a);
    vb = vld1q_s8(b);

    int32x4_t vc;
    vc = vmmlaq_s32(vc, va, vb);

    // Matrix multiply result = 780 1068 2092 2892
    // a[1] * b[1], a[1] * b[2],
    // a[2] * b[1], a[2] * b[2]
    printf("Matrix multiply result = %d %d %d %d \n",vc[0],vc[1],vc[2],vc[3]);
}

void test_dot() {
    // store in row
    int8_t a1[8] = {1,2,3,4,5,6,7,8};
    int8_t a2[8] = {9,10,11,12,13,14,15,16};

    // store in columns
    int8_t b1[8] = {17,18,19,20,21,22,23,24};
    int8_t b2[8] = {25,26,27,28,29,30,31,32};

    int8x8_t va1, va2, vb1, vb2;

    va1 = vld1_s8(a1);
    va2 = vld1_s8(a2);
    vb1 = vld1_s8(b1);
    vb2 = vld1_s8(b2);

    int32x2_t vc1, vc2, vc3, vc4;
    vc1 = vdot_s32(vc1, va1, vb1);
    vc2 = vdot_s32(vc2, va1, vb2);
    vc3 = vdot_s32(vc3, va2, vb1);
    vc4 = vdot_s32(vc4, va2, vb2);

    // Dot product result = 780 1068 2092 2892
    printf("Dot product result = %d %d %d %d\n",vc1[0]+vc1[1],
                        vc2[0]+vc2[1],vc3[0]+vc3[1],vc4[0]+vc4[1]);
}



// 定义一个函数，输入 poly 数字，返回多项式字符串
char *poly_to_string(poly8_t p)
{
    // 定义一个缓冲区，用于存放多项式字符串
    char buffer[32];

    // 初始化缓冲区为空字符串
    buffer[0] = '\0';

    // 定义一个标志位，用于记录 poly 数字是否有非零位
    int flag = 0;

    // 遍历 poly 数字的每一位
    for (int i = 0; i < 8; i++)
    {
        // 如果 poly 数字的第 i 位为 1，那么表示有 x^i 这一项
        if (p & (1 << i))
        {
            // 如果之前已经有非零位，那么先加上一个加号
            if (flag)
            {
                strcat(buffer, " + ");
            }

            // 根据 i 的值，拼接相应的字符串到缓冲区
            switch (i)
            {
            case 0:
                strcat(buffer, "1");
                break;
            case 1:
                strcat(buffer, "x");
                break;
            default:
                strcat(buffer, "x^");
                char temp[4];
                sprintf(temp, "%d", i);
                strcat(buffer, temp);
                break;
            }

            // 将标志位设为 1，表示已经有非零位
            flag = 1;
        }
    }

    // 如果缓冲区为空字符串，那么表示 poly 数字为 0，返回 "0"
    if (buffer[0] == '\0')
    {
        return "0";
    }

    // 否则，复制缓冲区的内容到一个新分配的字符串，并返回它
    else
    {
        char *result = malloc(strlen(buffer) + 1);
        strcpy(result, buffer);
        return result;
    }
}

void test_polynomial() {
    // 定义两个 poly8_t 类型的数组
    poly8_t a[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    poly8_t b[8] = {0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10};

    // 使用 vld1_p8 函数来加载两个 poly8x8_t 类型的向量
    poly8x8_t va = vld1_p8(a);
    poly8x8_t vb = vld1_p8(b);

    // 使用 vmul_p8 指令进行多项式乘法
    poly8x8_t vc = vmul_p8(va, vb);

    // 打印结果
    for (int i = 0; i < 8; i++)
    {
        printf("0x%02x ", vc[i]);
    }
    printf("\n");

    printf("Hex to Polynomial: \n");
    for (int i = 0; i < 8; i++)
    {
        printf("[(%s) * (%s) = %s]\n",poly_to_string(va[i]),poly_to_string(vb[i]), poly_to_string(vc[i]));
    }
    printf("\n");
}


void test_pair_across() {
    int a[4] = {1,2,3,4};
    int b[4] = {5,6,7,8};

    int32x4_t va,vb,vc;

    va = vld1q_s32(a);
    vb = vld1q_s32(b);

    vc = vpaddq_s32(va, vb);
    // pair add result 3 7 11 15 
    printf("pair add result %d %d %d %d \n",vc[0],vc[1],vc[2],vc[3]);


    // Across vector arithmetic
    // reduce sum
    int sum = vaddvq_s32(va);
    int64_t sum_widen = vaddlvq_s32(va);
    printf("sum = %d sum_widen = %ld \n",sum, sum_widen);
}


void test_table_lookup() {
    int8_t src[8] = {1,2,3,4,5,6,7,8};
    int8_t idx[8] = {1,1,1,3,3,3,6,6};

    int8x8_t v_src = vld1_s8(src);
    int8x8_t v_idx = vld1_s8(idx);

    // 2 2 2 4 4 4 7 7 
    int8x8_t v_dst = vtbl1_s8(v_src, v_idx);
    for (int i = 0; i < 8; i++)
    {
        printf("%d ", v_dst[i]);
    }
    printf("\n");
}


void test_vetor_manip() {
    int8_t src[8] = {1,2,3,4,5,6,7,8};
    int8_t idx[8] = {-1,-2,-3,-4,-5,-6,-7,-8};

    int8x8_t v_src = vld1_s8(src);
    int8x8_t v_idx = vld1_s8(idx);

    // 1 2 3 -4 5 6 7 8
    int8x8_t v_dst = vcopy_lane_s8(v_src, 3, v_idx, 3);
    for (int i = 0; i < 8; i++)
    {
        printf("%d ", v_dst[i]);
    }
    printf("\n");


    // -128 64 -64 32 -96 96 -32 16
    v_dst = vrbit_s8(v_src);
    for (int i = 0; i < 8; i++)
    {
        printf("%d ", v_dst[i]);
    }
    printf("\n");

    // create a vector with a 64bit input value
    v_dst = vcreate_s8(0x010203040a0b0c0d);
    // 13 12 11 10 4 3 2 1 
    for (int i = 0; i < 8; i++)
    {
        printf("%d ", v_dst[i]);
    }
    printf("\n");

    // 2 3 4 5 6 7 8 -1
    v_dst = vext_s8(v_src, v_idx, 1);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");
    // 4 5 6 7 8 -1 -2 -3
    v_dst = vext_s8(v_src, v_idx, 3);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");
    // -3 -2 -1 8 7 6 5 4
    v_dst = vrev64_s8(v_dst);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");

    printf("vector zip \n");
    //zip
    // 1 -1 2 -2 3 -3 4 -4 
    v_dst = vzip1_s8(v_src, v_idx);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");
    // 5 -5 6 -6 7 -7 8 -8
    v_dst = vzip2_s8(v_src, v_idx);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");

    printf("vector unzip \n");
    // unzip
    // 1 3 5 7 -1 -3 -5 -7 
    v_dst = vuzp1_s8(v_src, v_idx);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");
    // 2 4 6 8 -2 -4 -6 -8 
    v_dst = vuzp2_s8(v_src, v_idx);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");

    printf("vector transpose \n");
    // 1 3 5 7 -1 -3 -5 -7 
    v_dst = vtrn1_s8(v_src, v_idx);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");
    // 2 -2 4 -4 6 -6 8 -8 
    v_dst = vtrn2_s8(v_src, v_idx);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");

    printf("set vector lane \n");
    // 100 -2 4 -4 6 -6 8 -8 
    v_dst = vset_lane_s8(100, v_dst, 0);
    for (int i = 0; i < 8; i++) printf("%d ", v_dst[i]); printf("\n");
}


int main() {

    float32_t a[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float32_t b[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float32_t c[9] = {0};

    neon_add(a, b, c, 9);

    for (int i=0; i<9; i++) {
	    printf("%f ", c[i]);
    }
    printf("\n");

    load_lane_1();


    float32x4x2_t result = vld2q_f32(a);
    printf("%f %f %f %f %f %f %f %f \n",
            result.val[0][0], result.val[0][1], 
            result.val[1][0], result.val[1][1], 
            result.val[0][2], result.val[0][3], 
            result.val[1][2], result.val[1][3]);


    // 读2个数，重复放到2个vector的所有lane里
    float32x4x2_t result_dup = vld2q_dup_f32(a);
    // 1.000000 1.000000 2.000000 2.000000 1.000000 1.000000 2.000000 2.000000 
    printf("%f %f %f %f %f %f %f %f \n",
            result_dup.val[0][0], result_dup.val[0][1], 
            result_dup.val[1][0], result_dup.val[1][1], 
            result_dup.val[0][2], result_dup.val[0][3], 
            result_dup.val[1][2], result_dup.val[1][3]);


    float f16[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float32x4x4_t result_4 = vld4q_f32(f16);
    // 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 7.000000 8.000000
    // 9.000000 10.000000 11.000000 12.000000 13.000000 14.000000 15.000000 16.000000 
    printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n",
            result_4.val[0][0], result_4.val[1][0], 
            result_4.val[2][0], result_4.val[3][0], 
            result_4.val[0][1], result_4.val[1][1], 
            result_4.val[2][1], result_4.val[3][1],
            result_4.val[0][2], result_4.val[1][2], 
            result_4.val[2][2], result_4.val[3][2], 
            result_4.val[0][3], result_4.val[1][3], 
            result_4.val[2][3], result_4.val[3][3]);


    test_complex();
    test_complex_mac();
    test_complex_mac_270();

    //test_aes();

    test_matrix();
    test_dot();

    test_polynomial();

    test_pair_across();

    test_table_lookup();

    test_vetor_manip();

    return 0;
}
