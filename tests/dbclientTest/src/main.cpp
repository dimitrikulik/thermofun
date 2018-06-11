#include <iostream>
#include "ThermoFun.h"
#include "jsonio/io_settings.h"
#include "jsonio/nservice.h"

#include "DBClient/DatabaseClient.h"
#include "DBClient/ReactionData.h"
#include "DBClient/AbstractData.h"
#include "DBClient/SubstanceData.h"
#include "DBClient/ReactionSetData.h"
#include "DBClient/ThermoSetData.h"
#include "Common/ParseBsonTraversalData.h"

#include <sys/time.h>

using namespace std;
using namespace ThermoFun;

struct timeval st, en;

int main(int argc, char *argv[])
{

    cout << "Hello World!" << endl;
    gettimeofday(&st, NULL);
    jsonio::JsonioSettings::settingsFileName = "./Resources/ThermoFunDemoGUI.json";
    DatabaseClient dbc_;


    /// access to substance records
    auto results =  dbc_.substData().querySolvents(20);
    for( auto rez: results[0] )
        std::cout << rez << std::endl;


    auto results2 =  dbc_.substData().selectGiven(  {"thermodatasets/Aq17_Na_Cl_Si_2_0"}, false );
    for( auto rezt: results2 )
        std::cout << rezt <<  std::endl;

    auto results3 =  dbc_.substData().selectGiven(  {"thermodatasets/Aq17_Na_Cl_K_LMA_1_0"}, false );
    for( auto rezt: results2 )
        std::cout << rezt <<  std::endl;

    /// access to reactionSet records
    //ThermoFun::ThermoSetData thermoSetData(&datab);
    jsonio::GraphElementFunction afunc =  []( bool isVertex, const string& data )
    {
        if(isVertex )
            cout << "Vertex: ";
        else
            cout << "Edge: ";
        cout << jsonio::extractStringField( "_id", data ) << endl;
    };
    dbc_.thermoDataSet().traverceVertexes( "thermodatasets/Aq17_Na_Cl_K_LMA_1_0", afunc );

    auto list2 = dbc_.TraverseAllIncomingEdges("thermodatasets/Aq17_Na_Cl_K_LMA_1_0");
    Database db2 = databaseFromRecordList(dbc_, list2);

    auto availableElementskey = dbc_.availableElementsKey(20);

    auto substance = dbc_.substData( );

    availableElementskey.erase (availableElementskey.begin(),availableElementskey.begin()+1);

    auto substances = substance.selectGiven({20}, availableElementskey, true);

    auto reactions = dbc_.reactData().selectGiven({20}, {"H2O@", "Wollastonite", "Ca+2", "OH-", "SiO2@", "Anorthite", "Al+3"});

    auto reacSets = dbc_.reactSetData().selectGivenSubstances({20}, {"K+","Na+","Cl-","KCl@","Halite","Sylvite","NaCl@" });

    auto reacSets2 = dbc_.reactSetData().selectGiven({20}, {"KCl@","Halite","Sylvite","NaCl@"});

//    try {
//        auto list = dbc_.TraverseAllIncomingEdges("thermodatasets/Aq17_2_1");
//        Database db2 = databaseFromRecordList(dbc_, list);

//        for( auto row: list)
//            std::cout << row.first << " " << row.second << endl;
//    }
//    catch (jsonio::jsonio_exeption e) {}

//    try {
//        auto list_test = dbc_.recordsFromThermoDataSet("Aq17");
//    }
//    catch (jsonio::jsonio_exeption e) {}

    try {
        auto rec = dbc_.substData().loadRecord( "substances/AlOH+2_4_20", {"_id", "_label", "properties.formula", "properties.symbol"} );
        auto rec2 = dbc_.substData().loadRecord( "substances/Diaspore_3_20", {"_id", "_label", "properties.formula", "properties.symbol"} );

        auto a = 1;

//        if ("rec[3][]" != "AlOH2+")
//        {}
    }
    catch (jsonio::jsonio_exeption e) {}

    try {
        auto availableSourceTBs = dbc_.sourcetdbNamesIndexes();
        auto avaialbleElements = dbc_.availableElements(20);
        for( auto el:avaialbleElements )
            std::cout << el << "  " ;
        std::cout << "avaialbleElements" << endl;
        auto availableElementskey = dbc_.availableElementsKey(20);
        for( auto el:availableElementskey )
            std::cout << el.symbol << "  " ;
        std::cout << "availableElementskey" << endl;
        auto a = 1;
    }
    catch (jsonio::jsonio_exeption e) {}

    std::map<Element, double> elem = dbc_.parseSubstanceFormula("Mg4Al8Si2O20");
/*
    try {
        Database db = dbc_.thermoFunDatabase(20);

        Substance water;
        water.setName("water");
        water.setSymbol("H2O@_");
        water.setFormula("H2O");
        water.setSubstanceClass(SubstanceClass::type::AQSOLVENT);
        water.setAggregateState(AggregateState::type::AQUEOUS);

        water.setMethodGenEoS(MethodGenEoS_Thrift::type::CTPM_WJNR);

        water.setMethod_T(MethodCorrT_Thrift::type::CTM_WAT);

        db.addSubstance(water);

        Thermo th(db); //DBClient("./Resources/ThermoFun.ini").getDatabase(15)

        th.setSolventSymbol("H2O@_");

        double T = 650;
        double P = 2000;

        auto qtz = th.thermoPropertiesSubstance(T, P, "Quartz");

        auto watP = th.propertiesSolvent(T, P, "H2O@_");

        auto wat = th.thermoPropertiesSubstance(T, P, "H2O@_");

        auto ca = th.thermoPropertiesSubstance(T, P, "Ca+2");

        auto co2 = th.thermoPropertiesSubstance(T, P, "CO2@");

        ThermoPropertiesSubstance MgSi, CaSi, FeHSi, RaC, RaS, SiO, CaSi_FM, SiOaq;

    //    for (uint i = 0; i <150000; i++)
    //    {
    //        SiOaq = th.thermoPropertiesSubstance(T, P, "SiO2@");
    //    }

    //    for (uint i = 0; i <150000; i++)
    //    {
    ////        SiOaq = th.thermoPropertiesSubstanceF(T, P, "SiO2@");
    //    }

    //    for (uint i = 0; i <150000; i++)
    //    {
    //        SiOaq = th.thermoPropertiesSubstance(T, P, "SiO2@");
    //    }

        MgSi = th.thermoPropertiesSubstance(T, P, "MgSiO3@");

        CaSi = th.thermoPropertiesSubstance(T, P, "CaSiO3@");

        FeHSi = th.thermoPropertiesSubstance(T, P, "FeHSiO3+2");

        RaC = th.thermoPropertiesSubstance(T, P, "RaCO3");

        RaS = th.thermoPropertiesSubstance(T, P, "RaSO4");

        SiO = th.thermoPropertiesSubstance(T, P, "SiO3-2");

        CaSi = th.thermoPropertiesSubstance(T, P, "CaSiO3@_FM_test");
    }
    catch (jsonio::jsonio_exeption e) {}
*/

//    auto rcd = dbc_.reactData();

    auto rcd = dbc_.reactData();
    jsonio::DBQueryData query("{ \"_label\" : \"reaction\"}",jsonio::DBQueryData::qTemplate );
    auto loadedReacData = rcd.loadRecordsValues(query, 20, dbc_.availableElementsKey(20));

    auto list = dbc_.TraverseAllIncomingEdges("thermodatasets/Aq17_2_1");
    Database db = databaseFromRecordList(dbc_, list);


    ThermoFun::Interface interface (db);
    interface.setSolventSymbol("H2O@");

    ThermoFun::OutputSettings op;
    op.isFixed = true;
    op.outSolventProp       = true;
    op.calcReactFromSubst   = false;
    op.calcSubstFromReact   = false;
    interface.setOutputSettings(op);

    interface.setPropertiesUnits({"temperature", "pressure"},{"degC","bar"});
    interface.setPropertiesDigits({"reaction_gibbs_energy","reaction_entropy", "reaction_volume",
                                   "reaction_enthalpy","logKr", "temperature", "pressure"}, {0, 4, 4, 4, 4, 0, 0});

    interface.thermoPropertiesReaction({{25,1}}, {"NaCl@", "H2O@"}, {"reaction_gibbs_energy", "reaction_entropy",
                                                             "reaction_volume", "reaction_enthalpy", "logKr"}/*, calcReactFromSubst*/).toCSV("NaCl@test.csv");

    interface.thermoPropertiesReaction({0, 350, 25},{0,0,0},{"NaCl@","H2O@"}, {"reaction_gibbs_energy", "reaction_entropy",
                                                         "reaction_volume", "reaction_enthalpy", "logKr"}/*, calcReactFromSubst*/).toCSV("NaClPsat@test.csv");

    cout << "Bye World!" << endl;
    return 0;
}

