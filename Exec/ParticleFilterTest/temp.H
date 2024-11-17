class myParticle {
    public:
        double x, y, z, vx, vy, vz;
};
extern FILE *file_lightcone_csv;
extern std::vector<myParticle> shell_particles;

template<typename T>
inline void SwapEnd(T& var)
{
  char* varArray = reinterpret_cast<char*>(&var);
  for(long i = 0; i < static_cast<long>(sizeof(var)/2); i++)
    std::swap(varArray[sizeof(var) - 1 - i],varArray[i]);
}

void writeBinaryVTK(const std::string& filename,
                    const std::vector<myParticle>& shell_particles);
void writeBinarySimple(const std::string& filename,
                    const std::vector<myParticle>& shell_particles);
