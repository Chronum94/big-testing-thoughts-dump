module LatticeUnits
using Unitful; 
using PhysicalConstants.CODATA2018; 

length_scale = 0.1u"nm"
mass_scale = 
@unit phe "phe" PhononEnergy (PlanckConstant / (2pi))^2 / ((length_scale)^2 * AtomicMassConstant) false;
@unit phf "phf" PhononEfield phe / (ElementaryCharge * length_scale) false;
end

Unitful.register(LatticeUnits)

print(u"eV"(1.0u"phe"))
print(u"V/m"(1.0u"phf"))
