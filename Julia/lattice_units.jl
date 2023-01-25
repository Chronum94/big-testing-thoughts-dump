module LatticeUnits
using Unitful; 
using PhysicalConstants.CODATA2018; 
@unit phe "phe" PhononEnergy (PlanckConstant / (2pi))^2 / ((0.1u"nm")^2 * AtomicMassConstant) false;
end

Unitful.register(LatticeUnits)
print(u"eV"(1.0u"phe"))
