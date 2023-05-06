#ifndef __VEC3_H
#define __VEC3_H

#include <stdint.h>
#include <ostream>

// creates the Vec3 type
struct Vec3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    Vec3() {
        x = 0;
        y = 0;
        z = 0;
    }

    Vec3(uint32_t i, uint32_t j, uint32_t k) {
        x = i;
        y = j;
        z = k;
    }

    bool operator==(const Vec3 &v) const {
        return (x == v.x && y == v.y && z == v.z);
    }

    bool operator<(const Vec3 &v) const {
        return x < v.x
     || (x == v.x && y < v.y)
     || (x == v.x && y == v.y && z < v.z);
    }

    friend std::ostream& operator<<(std::ostream &out, Vec3 const& v) {
        out << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return out;
    }
};

#endif
