#ifndef __VEC4_H
#define __VEC4_H

#include <stdint.h>
#include <ostream>

// creates the Vec4 type to allow neighbor storage
struct Vec4 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t n;

    Vec4() {
        x = 0;
        y = 0;
        z = 0;
        n = 0;
    }

    Vec4(uint32_t i, uint32_t j, uint32_t k, uint32_t m) {
        x = i;
        y = j;
        z = k;
        n = m;
    }

    bool operator==(const Vec4 &v) const {
        return (x == v.x && y == v.y && z == v.z && n == v.n);
    }

    bool operator<(const Vec4 &v) const {
        return x < v.x
     || (x == v.x && y < v.y)
     || (x == v.x && y == v.y && z < v.z)
     || (x == v.x && y == v.y && z == v.z && n < v.n);
    }

    friend std::ostream& operator<<(std::ostream &out, Vec4 const& v) {
        out << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.n << ")";
        return out;
    }
};

#endif
