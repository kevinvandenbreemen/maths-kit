package com.vandenbreemen.linalg.api;

/**
 * Some kind of vector
 */
public interface Vector {
    int length();

    double entry(int position);

    default String asString(){
        StringBuilder bld = new StringBuilder(getClass().getSimpleName()).append(":  {");
        for(int i=0; i<length(); i++){
            bld.append(entry(i));
            if(i < length()-1){
                bld.append(", ");
            }
        }
        bld.append(" }");

        return bld.toString();
    }
}
